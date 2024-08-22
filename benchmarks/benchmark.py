import torch
import os
import bmtrain as bmt
import jsonlines as jl
from utils import write_res, generate_inp, backward, ref_attn, flash,  ring
from burst_attn.comm import get_world_size, print_rank, get_rank
import math
import torch.utils.benchmark as benchmark
import torch.distributed as dist
from burst_attn import burst_attn_func, burst_attn_func_striped
import numpy as np

setting = {}
num_iter = 50


def flops(batch, seqlen, nheads, headdim, causal,  mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def init_setting():
    setting["batch_size"] = [5]
    # 8k each gpu and 4k each gpu
    setting["seqlen"] = [1024 * 4 * get_world_size(), 1024 * 8 * get_world_size()]
    # setting["seqlen"] = [1024 * 4 * get_world_size()]
    setting["num_heads"] = [32]
    setting["dim"] = [128]
    setting["causal"] = [True, False]
    setting["double_ring"] = [True, False]
    setting["bwd_opt"] = [True, False]



def get_setting():
    if os.environ["LOCAL_WORLD_SIZE"] == os.environ["WORLD_SIZE"]:
        double_ring_setting = [False]
        if get_rank() == 0:
            print("Local node, disable double_ring")
    else:
        double_ring_setting = setting["double_ring"]
    for batch_size in setting["batch_size"]:
        for seqlen in setting["seqlen"]:
            for num_heads in setting["num_heads"]:
                for dim in setting["dim"]:
                    for causal in setting["causal"]:
                        for bwd_opt in setting['bwd_opt']:
                            for double_ring in double_ring_setting:
                                yield (
                                    batch_size,
                                    seqlen,
                                    num_heads,
                                    dim,
                                    causal,
                                    double_ring,
                                    bwd_opt
                                )


mapping = {
    "flash": flash,
    "normal": ref_attn,
    "ring": ring,
}


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    desc="",
    verbose=False,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
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
    b, seqlen, n, d, causal, double_ring, opt_bwd = settings

    if method == "flash":
        shape = (b, seqlen, n, d)
    elif method.startswith("burst"):
        shape = (b, seqlen//get_world_size(), n, d)
    elif method in ["normal", "ring"]:
        if method == "ring":
            s_per_device = seqlen // get_world_size()
        shape = (b, n, s_per_device, d)

    torch.cuda.synchronize()
    forward_func = mapping[method]
    inp = generate_inp(*shape)
    for _ in range(num_iter):
        if method in ["flash"]:
            forward_func(*inp, causal=causal)
        elif method.startswith("burst"):
            forward_func(*inp, causal=causal, double_ring=double_ring, opt_bwd=opt_bwd)
        else:
            forward_func(*inp)
    # print_rank("warmup")
    kwargs = {}
    if method in ["burst", "burst_striped"]:
        kwargs["double_ring"] = double_ring
        kwargs['opt_bwd'] = opt_bwd
    if method in ["flash", "burst", "burst_striped"]:
        _, forward_time = benchmark_forward(
            forward_func,
            *inp,
            causal=causal,
            desc=method,
            verbose=False,
            repeats=num_iter,
            **kwargs,
        )
        _, backward_time = benchmark_backward(
            forward_func,
            *inp,
            causal=causal,
            desc=method,
            verbose=False,
            repeats=num_iter,
            **kwargs
        )
    else:
        _, forward_time = benchmark_forward(
            forward_func, *inp, desc=method, verbose=False, repeats=num_iter
        )
        _, backward_time = benchmark_backward(
            forward_func, *inp, desc=method, verbose=False, repeats=num_iter
        )
    forward_time = forward_time.mean
    backward_time = backward_time.mean
    forward_backward_time = forward_time + backward_time
    title = "batch_size|seqlen|num_heads|dim|causal|double_ring|opt_bwd"
    head = "|".join([str(x) for x in settings])
    head = "|".join("{:^16s}".format(x) for x in head.split("|"))
    title = "|".join("{:^16s}".format(x) for x in title.split("|"))
    head = title + "\n\t" + head
    ratio = get_world_size() if method != "flash" else 1
    fwd_tflops = efficiency(flops(b, seqlen, n, d, causal, mode="fwd"), forward_time) / ratio
    fwd_bwd_tflops = (
        efficiency(flops(b, seqlen, n, d, causal, mode="fwd_bwd"), forward_backward_time) / ratio
    )
    bwd_tflops = efficiency(flops(b, seqlen, n, d, causal, mode="bwd"), backward_time) / ratio
    print_rank(
        f"{method}\n\t{head}\n\tTime: | forward: {forward_time:.2f} s | forward_backward: {forward_backward_time:.2f} s"
    )
    print_rank(
        f"\tTFLOPS| | forward: {fwd_tflops:.2f} TFLOPS | forward_backward: {fwd_bwd_tflops:.2f} TFLOPS | backward: {bwd_tflops:.2f} TFLOPS"
    )
    return forward_time, forward_backward_time


def local_global_mesh(group, local_size):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_ranks = [
        [i for i in range(start, start + local_size)]
        for start in range(0, world_size, local_size)
    ]
    global_ranks = [
        [i for i in range(start, world_size, local_size)] for start in range(local_size)
    ]
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

def get_burst_func(is_striped=False):
    all_group = get_group()
    def wrapper(
        q,
        k,
        v,
        group=None,
        causal=False,
        opt_bwd=True,
        deterministic=False,
        double_ring=False,
    ):
        if double_ring:
            group, intra_g, inter_g = all_group
        else:
            group, intra_g, inter_g = None, None, None
        func = burst_attn_func if not is_striped else burst_attn_func_striped
        res_burst = func(
            q,
            k,
            v,
            None,
            "cuda",
            causal,
            opt_bwd,
            deterministic,
            group,
            [intra_g, inter_g],
        )
        return res_burst
    return wrapper

def run_bench_torch():
    # bmt.init_distributed()
    group = dist.init_process_group(backend="nccl")
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = dist.get_world_size()
    init_setting()
    burst_func = get_burst_func()
    burst_striped = get_burst_func(True)
    mapping["burst"] = burst_func
    mapping["burst_striped"] = burst_striped

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    fi = jl.open("results_torch.jsonl", "a")
    for i, setting in enumerate(get_setting()):
        b, s, n, d, causal, double_ring, bwd_opt = setting
        for method in [
            "burst_striped",
            "burst",
            "flash",
        ]:
            if method == "burst_striped" and not causal:
                continue

            f, fb = benchmark_one_setting(method, setting)
            write_res(*setting, method, f, fb, fi)
    fi.close()


def get_group(create_dq_group=True):
    # if dq use the same group with q, g, lse, softmax_d, it wont pass the correctness pass in double_ring
    if bmt.init.is_initialized():
        return (
            bmt.config["comm"],
            (bmt.config["local_comm"], bmt.config["local_comm2"]),
            (bmt.config["local_idx_comm"], bmt.config["local_idx_comm2"]),
        )
    else:
        nnodes = int(os.environ["WORLD_SIZE"]) // int(os.environ["LOCAL_WORLD_SIZE"])
        local_size = get_world_size() // nnodes
        group_ranks = np.array(list(range(get_world_size())))
        intra_ranks = group_ranks.reshape(-1, local_size)
        inter_ranks = intra_ranks.transpose()
        intra_group, _ = torch.distributed.new_subgroups_by_enumeration(
            intra_ranks.tolist(), backend="nccl"
        )
        inter_group, _ = torch.distributed.new_subgroups_by_enumeration(
            inter_ranks.tolist(), backend="nccl"
        )
        if create_dq_group:
            intra_group2, _ = torch.distributed.new_subgroups_by_enumeration(
                intra_ranks.tolist(), backend="nccl"
            )
            inter_group2, _ = torch.distributed.new_subgroups_by_enumeration(
                inter_ranks.tolist(), backend="nccl"
            )
        if not create_dq_group:
            return None, intra_group, inter_group
        else:
            return (
                None,
                (
                    intra_group,
                    intra_group2,
                ),
                (inter_group, inter_group2),
            )


def run_bench_bmt():
    bmt.init_distributed()
    init_setting()
    bmt.config["sp_stream"] = torch.cuda.Stream(-1)
    bmt.config["sp_stream2"] = torch.cuda.Stream(-1)
    bmt.config["sp_stream3"] = torch.cuda.Stream(-1)
    bmt.config["sp_stream4"] = torch.cuda.Stream(-1)
    fi = jl.open("results_bmt.jsonl", "a")
    burst_func = get_burst_func()
    burst_striped = get_burst_func(True)

    mapping["burst"] = burst_func
    mapping["burst_striped"] = burst_striped
    for i, settings in enumerate(get_setting()):
        b, s, n, d, causal, double_ring, bwd_opt = settings
        for method in ["burst", "flash", "burst_striped"]:
            if method == "burst_striped" and not causal:
                continue
            f, fb = benchmark_one_setting(method, settings)
            write_res(*settings, method, f, fb, fi)
    fi.close()


if __name__ == "__main__":
    # run_bench_bmt()
    run_bench_torch()
