import torch
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
from flash_triton import flash_attn_func as flash_triton
import bmtrain as bmt
import jsonlines as jl
from burst_attn import OpBurstAttn
from ring_attn import ring_attn
setting = {}
num_iter = 100

def init_setting(): 
    setting['batch_size'] = [1]
    setting['seqlen'] = [65536, 131072, 262144, 524288, 1048576]
    setting['num_heads'] = [32]
    setting['dim'] = [128]
    init_bmt()
def init_bmt():
    bmt.init_distributed()
    bmt.config['sp_stream'] = torch.cuda.Stream(-1)

def get_setting():
    for batch_size in setting['batch_size']:
        for seqlen in setting['seqlen']:
            for num_heads in setting['num_heads']:
                for dim in setting['dim']:
                    yield batch_size, seqlen, num_heads, dim

def generate_inp(*shape):
    qkv = [torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)]
    return qkv

def backward(output, qkv):
    g = torch.randn_like(output)
    torch.autograd.grad(output, qkv, g)
    

def ref_attn(q, k, v):
    scale = q.shape[-1] ** -0.5
    s = q @ k.transpose(-2, -1) * scale
    s = torch.softmax(s, dim=-1)
    p = s @ v
    return p

def flash(q, k, v):
    scale = q.shape[-1] ** -0.5
    batch_size,seqlen,_,_ = q.shape
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=q.device)
    return flash_cuda(q,k,v,causal=False,softmax_scale=None)

def burst(q, k, v):
    res_burst = OpBurstAttn.apply(q, k, v, None, "cuda", True)
    return res_burst

def ring(q, k ,v):
    res_ring = ring_attn(q,k,v)
    return res_ring

def write_res(b,s,n,d,m,f,fb,file):
    item = {
        "batch_size":b,
        "seqlen":s,
        "num_heads":n,
        "dim":d,
        "method":m,
        "forward":f,
        "forward_backward":fb
    }
    if bmt.rank() == 0:
        file.write(item)
    
mapping = {
    "burst": burst,
    "flash": flash,
    "normal": ref_attn,
    "ring": ring
}
def benchmark_one_setting(method, settings):
    b,s,n,d = settings
    if method in ['flash','burst']:
        if method == "burst":
            s = s // bmt.world_size()
        shape = (b, s, n, d)
    elif method in [ 'normal','ring']:
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
    bmt.print_rank(f"{method}| {s} | forward: {forward_time} ms | forward_backward: {forward_backward_time} ms")
    return forward_time, forward_backward_time

def run_bench():
    init_setting()
    fi = jl.open("results.jsonl", "a")
    for i,s in enumerate(get_setting()):
        #for method in ['burst', 'ring', 'flash',]:
        for method in ['burst']:
            # if ((i+1) % 4) == bmt.rank():
            f,fb = benchmark_one_setting(method, s)
            write_res(*s, method, f, fb, fi)
    fi.close()
    
if __name__ == "__main__":
    run_bench()
