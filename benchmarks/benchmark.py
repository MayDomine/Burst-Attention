import torch
import bmtrain as bmt
import jsonlines as jl
from utils import write_res, generate_inp, backward, ref_attn, flash, burst, ring
from burst_attn.comm import get_world_size, print_rank, get_rank

setting = {}
num_iter = 5


def init_setting():
    setting["batch_size"] = [1]
    setting["seqlen"] = [131072]
    setting["num_heads"] = [32]
    setting["dim"] = [64]
    setting["causal"] = [True, False]


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


def benchmark_one_setting(method, settings):
    b, s, n, d, causal = settings
    if method in ["flash", "burst"]:
        if method in ["burst"]:
            s = s // get_world_size()
        shape = (b, s, n, d)
    elif method in ["normal", "ring"]:
        if method == "ring":
            s = s // get_world_size()
        shape = (b, n, s, d)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    forward_func = mapping[method]
    inp = generate_inp(*shape)

    for _ in range(num_iter):
        if method in ["flash", "burst"]:
            forward_func(*inp, causal=causal)
        else:
            forward_func(*inp)
    print_rank("warmup")
    start.record()
    for _ in range(num_iter):
        with torch.no_grad():
            forward_func(*inp)
    end.record()
    torch.cuda.synchronize()

    forward_time = start.elapsed_time(end)
    inp = generate_inp(*shape)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iter):
        out = forward_func(*inp)
        backward(out, inp)
    end.record()
    torch.cuda.synchronize()
    forward_backward_time = start.elapsed_time(end)

    s = "|".join([str(x) for x in settings])
    forward_time = forward_time / num_iter * 100
    forward_backward_time = forward_backward_time / num_iter * 100
    print_rank(
        f"{method}| {s} | forward: {forward_time} ms | forward_backward: {forward_backward_time} ms"
    )
    return forward_time, forward_backward_time


def run_bench_torch():
    # bmt.init_distributed()
    init_setting()
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
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
    for i, s in enumerate(get_setting()):
        for method in ["burst", "flash"]:
            f, fb = benchmark_one_setting(method, s)
            write_res(*s, method, f, fb, fi)
    fi.close()


if __name__ == "__main__":
    # run_bench_bmt()
    run_bench_torch()
