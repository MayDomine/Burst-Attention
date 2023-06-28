import torch

def benchmark_forward(func, args, desc=""):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    print(f"{desc} forward: {start.elapsed_time(end)} ms")