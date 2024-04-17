import torch
import bmtrain as bmt
import jsonlines as jl
from utils import write_res, generate_inp, backward, ref_attn, flash, burst, ring
from burst_attn.comm import init_comm_config

setting = {}
num_iter = 100


def init_setting(backend="bmt"):
    # sequence length benchmark
    setting["batch_size"] = [1]
    setting["seqlen"] = [4096]
    # setting['seqlen'] = [65536, 131072, 262144, 524288, 1048576]
    setting["num_heads"] = [32]
    setting["dim"] = [128]
    init_comm_config(backend)


def get_setting():
    for batch_size in setting["batch_size"]:
        for seqlen in setting["seqlen"]:
            for num_heads in setting["num_heads"]:
                for dim in setting["dim"]:
                    yield batch_size, seqlen, num_heads, dim


mapping = {
    "burst": burst,
    "flash": flash,
    "normal": ref_attn,
    "ring": ring,
}


def benchmark_one_setting(method, settings):
    b, s, n, d = settings
    if method in ["flash", "burst"]:
        if method in ["burst"]:
            s = s // bmt.world_size()
        shape = (b, s, n, d)
    elif method in ["normal", "ring"]:
        if method == "ring":
            s = s // bmt.world_size()
        shape = (b, n, s, d)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    forward_func = mapping[method]
    inp = generate_inp(*shape)
    start.record()
    for _ in range(num_iter):
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
    bmt.print_rank(
        f"{method}| {s} | forward: {forward_time} ms | forward_backward: {forward_backward_time} ms"
    )
    return forward_time, forward_backward_time


def run_bench():
    init_setting(backend="bmt")
    fi = jl.open("results.jsonl", "a")
    for i, s in enumerate(get_setting()):
        for method in ["burst", "ring", "flash"]:
            f, fb = benchmark_one_setting(method, s)
            write_res(*s, method, f, fb, fi)
    fi.close()


if __name__ == "__main__":
    run_bench()
