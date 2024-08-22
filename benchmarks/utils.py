import math
import torch
import bmtrain as bmt
from ring_attn import RingQK, RingAV
from burst_attn import burst_attn_func, burst_attn_func_striped
from burst_attn.comm import get_rank
import torch.distributed as dist
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda


def ring_attn(q, k, v, sm_scale=1.0):
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    sub_seq = q.shape[2]
    hidden_dim = q.shape[-1]
    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)
    attn_score = RingQK.apply(q, k, batch_size, num_heads, sub_seq, sm_scale)
    attn_score = torch.softmax(attn_score, dim=-1)
    out = RingAV.apply(attn_score, v, batch_size, num_heads, hidden_dim, sub_seq)
    return out


def flops(batch_size, seqlen, dim, num_heads, mode="fwd"):
    causal = False
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch_size * seqlen**2 * num_heads * dim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def generate_inp(*shape):
    qkv = [
        torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
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


def flash(q, k, v, causal=False):
    return flash_cuda(q, k, v, causal=causal, softmax_scale=None)


def burst(q, k, v, group=None, causal=False, opt_bwd=True, deterministic=False):
    res_burst = burst_attn_func(
        q, k, v, None, "cuda", causal, opt_bwd, deterministic, None
    )
    return res_burst


def ring(q, k, v):
    res_ring = ring_attn(q, k, v)
    return res_ring


def write_res(b, s, n, d, causal, double_ring, opt_bwd, method, foward_t, fb_t, file):
    item = {
        "batch_size": b,
        "seqlen": s,
        "num_heads": n,
        "double_ring": double_ring,
        "dim": d,
        "method": method,
        "forward": foward_t,
        "forward_backward": fb_t,
        "causal": causal,
    }
    if get_rank() == 0:
        file.write(item)
