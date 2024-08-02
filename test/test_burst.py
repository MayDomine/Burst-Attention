import sys
import torch
import os
import bmtrain as bmt

sys.path.append("..")
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
from burst_attn.comm import synchronize, get_rank, get_world_size
from burst_attn import OpBurstAttn
from checker import check_helper
from burst_attn import OpBurstAttn, OpBurstAttnCausal
from burst_attn.comm import Ring


def test_msg(test_func, msg, *args, **kwargs):
    try:
        test_func(*args, **kwargs)
        bmt.print_rank(msg, " Success")
    except:
        bmt.print_rank(msg, " Failed")
        exit()


def get_chunk(t, dim, half_reputation=False):
    if half_reputation:
        splits = t.chunk(get_world_size() * 2, dim=dim)
        partition1, partition2 = (
            splits[get_rank()],
            splits[get_world_size() * 2 - get_rank() - 1],
        )
        return torch.cat([partition1, partition2], dim=dim).contiguous()
    else:
        return t.chunk(get_world_size(), dim=dim)[get_rank()].contiguous()


def burst(q, k, v, group=None):
    res_burst = OpBurstAttn.apply(q, k, v, None, "cuda", False, group)
    return res_burst


def test(q, k, v, func, grad_output):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    grad_output = grad_output.contiguous()
    o = func(q, k, v)
    gq, gk, gv = torch.autograd.grad(o, (q, k, v), grad_output)
    return o, (gq, gk, gv)


def burst_func(q, k, v, causal=False):
    if causal:
        return OpBurstAttnCausal.apply(q, k, v, None, "cuda", True, None)
    else:
        return OpBurstAttn.apply(q, k, v, None, "cuda", True, None)

def test_ring_comm():
    comm = Ring(None)
    tensor1 = torch.ones(100, 100).cuda() * (get_rank() + 1)
    tensor2 = torch.zeros(100, 100).cuda()
    comm.ring_send_recv([tensor1], [tensor2])
    comm.commit()
    comm.wait()
    print(tensor2.max())

def test_burst(causal=False):
    print(f"Checking Burst Attn Causal = {causal}...")
    b, s, n, d = 2, 1024, 16, 32
    if get_rank() == 0:
        qkv = torch.randn(b, s * 3, n, d, dtype=torch.float16).cuda()
        grad_output = torch.randn(b, s, n, d, dtype=torch.float16).cuda()
        torch.save(qkv, "qkv.pt")
        torch.save(grad_output, "grad.pt")
    synchronize()
    flash = lambda q, k, v: flash_cuda(q, k, v, causal=False, softmax_scale=None)
    qkv = torch.load("qkv.pt", map_location="cuda", weights_only=True)
    grad_output = torch.load("grad.pt", map_location="cuda", weights_only=True)
    qkv1 = [t.clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]
    if get_rank() == 0:
        os.remove("qkv.pt")
        os.remove("grad.pt")

    o_ref, g_ref = test(qkv1[0], qkv1[1], qkv1[2], flash, grad_output)
    for i in range(3):
        qkv1[i] = get_chunk(qkv1[i], 1, causal)
        qkv1[i] = qkv1[i].clone().detach().requires_grad_()
    grad_output = get_chunk(grad_output, 1, causal)
    grad_output = (
        grad_output
        .clone()
        .detach()
        .contiguous()
    )
    qkv1 = [t.contiguous() for t in qkv1]
    o1 = burst_func(qkv1[0], qkv1[1], qkv1[2], causal)
    grad_qkv1 = torch.autograd.grad(o1, qkv1, grad_output)
    # o1, grad_qkv1 = test(qkv1[0], qkv1[1], qkv1[2], burst_func, grad_output)
    o1 = o1.contiguous()
    grad_qkv1 = [g.contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref, 1, causal)
    g_ref = [get_chunk(g, 1, causal) for g in g_ref]
    torch.cuda.synchronize()
    print((o_ref - o1).max())
    test_msg(check_helper, "Output Correctness Check", o_ref, o1)
    test_msg(check_helper, "Value Correctness Check", g_ref[2], grad_qkv1[2])
    test_msg(check_helper, "Key Correctness Check", g_ref[1], grad_qkv1[1])
    test_msg(check_helper, "Query Correctness Check", g_ref[0], grad_qkv1[0])


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(torch.distributed.get_rank())
    # bmt.init_distributed()
    # bmt.config['sp_stream'] = torch.cuda.Stream()
    test_burst(False)
    # test_ring_comm()
