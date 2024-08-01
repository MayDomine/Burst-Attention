import sys
import torch
import os
import bmtrain as bmt
sys.path.append("..")  
from benchmarks.utils import burst, ref_attn
from burst_attn.comm import synchronize, get_rank
from checker import check_helper


def test_msg(test_func, msg, *args, **kwargs):
    try:
        test_func(*args, **kwargs)
        bmt.print_rank(msg, " Success")
    except:
        bmt.print_rank(msg, " Failed")
        exit()


def get_chunk(t, dim):
    return t.chunk(bmt.world_size(), dim=dim)[bmt.rank()].contiguous()


def test(q, k, v, func, grad_output):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    grad_output = grad_output.contiguous()
    o = func(q, k, v)
    gq, gk, gv = torch.autograd.grad(o, (q, k, v), grad_output)
    return o, (gq, gk, gv)


def test_burst():
    b, s, n, d = 2, 1024, 16, 32
    if get_rank() == 0:
        qkv = torch.randn(b, s * 3, n, d, dtype=torch.float16).cuda()
        grad_output = torch.randn(b, s, n, d, dtype=torch.float16).cuda()
        torch.save(qkv, "qkv.pt")
        torch.save(grad_output, "grad.pt")
    synchronize()
    qkv = torch.load("qkv.pt", map_location="cuda")
    grad_output = torch.load("grad.pt", map_location="cuda")
    qkv1 = [t.clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]
    if get_rank() == 0:
        os.remove("qkv.pt")
        os.remove("grad.pt")

    o_ref, g_ref = test(qkv1[0], qkv1[1], qkv1[2], ref_attn, grad_output)
    for i in range(3):
        qkv1[i] = qkv1[i].chunk(bmt.world_size(), dim=2)[bmt.rank()]
        qkv1[i] = qkv1[i].transpose(1, 2).clone().detach().requires_grad_()
    grad_output = (
        grad_output.transpose(1, 2)
        .chunk(bmt.world_size(), dim=1)[bmt.rank()]
        .clone()
        .detach()
        .contiguous()
    )
    o1, grad_qkv1 = test(qkv1[0], qkv1[1], qkv1[2], burst, grad_output)
    o1 = o1.transpose(1, 2).contiguous()
    grad_qkv1 = [g.transpose(1, 2).contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref, dim=2)
    g_ref = [get_chunk(g, dim=2) for g in g_ref]
    test_msg(check_helper, "Output Correctness Check", o_ref, o1)
    test_msg(check_helper, "Value Correctness Check", g_ref[2], grad_qkv1[2])
    test_msg(check_helper, "Key Correctness Check", g_ref[1], grad_qkv1[1])
    test_msg(check_helper, "Query Correctness Check", g_ref[0], grad_qkv1[0])


if __name__ == "__main__":

    test_burst()
