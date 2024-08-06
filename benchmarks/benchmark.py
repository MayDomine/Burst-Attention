import torch
import os
import bmtrain as bmt
import jsonlines as jl
from utils import write_res, generate_inp, backward, ref_attn, flash, burst, ring
from burst_attn.comm import get_world_size, print_rank, get_rank
import math
import torch.utils.benchmark as benchmark
import torch.distributed as dist
from burst_attn import OpBurstAttn
setting = {}
num_iter = 50

def flops(batch, seqlen,  nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def init_setting():
    setting["batch_size"] = [1]
    setting["seqlen"] = [8192 * 16, 8192 * 32]
    setting["num_heads"] = [5]
    setting["dim"] = [128]
    setting["causal"] = [False]


def get_setting():
    for batch_size in setting["batch_size"]:
        for seqlen in setting["seqlen"]:
            for num_heads in setting["num_heads"]:
                for dim in setting["dim"]:
                    for causal in setting["causal"]:
                        yield batch_size, seqlen, num_heads, dim, causal


mapping = {
    "burst": burst,
    "flash": flash,
    "normal": ref_attn,
    "ring": ring,
}


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=False, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=False,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def benchmark_one_setting(method, settings):
    b, s, n, d, causal = settings
    if method in ["flash", "burst"] or method.startswith("burst"):
        if method in ["burst"]:
            s = s // get_world_size()
        shape = (b, s, n, d)
    elif method in ["normal", "ring"]:
        if method == "ring":
            s = s // get_world_size()
        shape = (b, n, s, d)
        
    torch.cuda.synchronize()
    forward_func = mapping[method]
    inp = generate_inp(*shape)
    for _ in range(num_iter):
        if method in ["flash", "burst"]:
            forward_func(*inp, causal=causal)
        else:
            forward_func(*inp)
    print_rank("warmup")
    if method in ["flash", "burst"]:
        _, forward_time = benchmark_forward(forward_func, *inp, causal=causal, desc=method, verbose=False, repeats=num_iter)
        _, backward_time = benchmark_backward(forward_func, *inp, causal=causal, desc=method, verbose=False, repeats=num_iter) 
    else:
        _, forward_time = benchmark_forward(forward_func, *inp, desc=method, verbose=False, repeats=num_iter)
        _, backward_time = benchmark_backward(forward_func, *inp, desc=method, verbose=False, repeats=num_iter)
    forward_time = forward_time.mean
    backward_time = backward_time.mean
    forward_backward_time = forward_time + backward_time
    s = "|".join([str(x) for x in settings])
    ratio = get_world_size() if method == "burst" else 1
    fwd_tflops = efficiency(flops(*settings,  mode="fwd"), forward_time) / ratio
    fwd_bwd_tflops = efficiency(flops(*settings, mode="fwd_bwd"), forward_backward_time) / ratio
    bwd_tflops = efficiency(flops(*settings, mode="bwd"), backward_time) / ratio
    print_rank(
        f"{method}| {s}\n\tTime: | forward: {forward_time:.2f} s | forward_backward: {forward_backward_time:.2f} s"
    )
    print_rank(
        f"\tTFLOPS| | forward: {fwd_tflops:.2f} TFLOPS | forward_backward: {fwd_bwd_tflops:.2f} TFLOPS | backward: {bwd_tflops:.2f} TFLOPS"
    )
    return forward_time, forward_backward_time

def local_global_mesh(group, local_size):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_ranks = [[i for i in range(start, start + local_size)] for start in range(0, world_size, local_size) ]
    global_ranks = [[i  for i in range(start, world_size, local_size)] for start in range(local_size)]
    group1 = None
    group2 = None
    for ranks in local_ranks:
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            group1 = group
    for ranks in global_ranks:
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            group2 = group 

    return group1, group2

def run_bench_torch():
    # bmt.init_distributed()
    init_setting()
    group = dist.init_process_group(backend="nccl")
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = dist.get_world_size()
    group1, group2 = local_global_mesh(group, local_world_size)
    def burst(q, k, v, group=None, causal=False, opt_bwd=True, deterministic=False):
        res_burst = OpBurstAttn.apply(
            q, k, v, None, "cuda", causal, opt_bwd, deterministic, group, [group1, group2]
        )
        return res_burst
    mapping["burst"] = burst
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    fi = jl.open("results_torch.jsonl", "a")
    for i, s in enumerate(get_setting()):
        for method in ["flash", "burst"]:
            f, fb = benchmark_one_setting(method, s)
            write_res(*s, method, f, fb, fi)
    fi.close()


def run_bench_bmt():
    init_setting()
    bmt.init_distributed()
    bmt.config["sp_stream"] = torch.cuda.Stream(-1)
    fi = jl.open("results_bmt.jsonl", "a")
    def burst(q, k, v, group=None, causal=False, opt_bwd=True, deterministic=False):
        res_burst = OpBurstAttn.apply(
            q, k, v, None, "cuda", causal, opt_bwd, deterministic, group, [bmt.config['local_comm'], bmt.config['local_idx_comm']]
        )
        return res_burst
    mapping['burst'] = burst
    for i, s in enumerate(get_setting()):
        for method in ["burst", "flash"]:
            f, fb = benchmark_one_setting(method, s)
            write_res(*s, method, f, fb, fi)
    fi.close()


if __name__ == "__main__":
    run_bench_bmt()
    # run_bench_torch()
