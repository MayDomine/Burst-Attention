import sys
import torch
import os
import bmtrain as bmt

sys.path.append("..")
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
from burst_attn.comm import synchronize, get_rank, get_world_size
from checker import check_helper
from burst_attn import OpBurstAttn
from burst_attn.comm import Ring, get_rank, get_world_size, print_rank, gather_obj


def test_msg(test_func, msg, *args, **kwargs):
    try:
        e = test_func(*args, **kwargs)
        succes = 1
    except Exception as e:
        succes = 0
    res = gather_obj(succes)
    if get_rank() == 0:
        if 0 in res:
            print(msg, f" Failed: {res.count(0)}/{len(res)} failed.")
            print("\tRank(s) that failed: ", [i for i, r in enumerate(res) if r == 0])
        else:
            print(msg, f" Success")


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


def burst(q, k, v, group=None, causal=False, opt_bwd=True, deterministic=True):
    res_burst = OpBurstAttn.apply(
        q, k, v, None, "cuda", causal, opt_bwd, deterministic, None
    )
    return res_burst


def test(q, k, v, func, grad_output):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    grad_output = grad_output.contiguous()
    o = func(q, k, v)
    gq, gk, gv = torch.autograd.grad(o, (q, k, v), grad_output)
    return o, (gq, gk, gv)


def burst_func(q, k, v, causal=False, optimize_bwd_comm=False, deterministic=False):
    return OpBurstAttn.apply(
        q, k, v, None, "cuda", causal, optimize_bwd_comm, deterministic, None
    )


def test_ring_comm():
    comm = Ring(None)
    t1 = torch.ones(100, 100).cuda() * (get_rank() + 1)
    t2 = torch.ones(100, 100).cuda() * (get_rank() + 1)
    r1 = torch.zeros(100, 100).cuda()
    r2 = torch.zeros(100, 100).cuda()
    comm._ring_send_recv_base([t1, t2], [r1, r2])
    comm.commit()
    comm.wait()


def test_burst(causal=False, optimize_bwd_comm=False, deterministic=True):
    print_rank(
        f"Checking Burst Attn Causal = {causal}... Optimize Bwd Comm = {optimize_bwd_comm}... Deterministic = {deterministic}..."
    )
    b, s, n, d = 2, 2048, 16, 32
    if get_rank() == 0:
        qkv = torch.randn(b, s * 3, n, d, dtype=torch.float16).cuda()
        grad_output = torch.randn(b, s, n, d, dtype=torch.float16).cuda()
        torch.save(qkv, "qkv.pt")
        torch.save(grad_output, "grad.pt")
    synchronize()
    flash = lambda q, k, v: flash_cuda(q, k, v, causal=causal, softmax_scale=None)
    qkv = torch.load("qkv.pt", map_location="cuda", weights_only=True)
    grad_output = torch.load("grad.pt", map_location="cuda", weights_only=True)
    qkv1 = [t.clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]
    qkv1_buf = [None] * 3
    for i in range(3):
        qkv1_buf[i] = get_chunk(qkv1[i], 1, causal).detach().clone().requires_grad_()
    if get_rank() == 0:
        os.remove("qkv.pt")
        os.remove("grad.pt")

    o_ref, g_ref = test(qkv1[0], qkv1[1], qkv1[2], flash, grad_output)
    grad_output = get_chunk(grad_output, 1, causal)
    grad_output = grad_output.clone().detach().contiguous()
    for i in range(3):
        qkv1[i] = qkv1_buf[i]
    qkv1 = [t.contiguous() for t in qkv1]
    o1 = burst_func(qkv1[0], qkv1[1], qkv1[2], causal, optimize_bwd_comm, deterministic)
    grad_qkv1 = torch.autograd.grad(o1, qkv1, grad_output)
    # o1, grad_qkv1 = test(qkv1[0], qkv1[1], qkv1[2], burst_func, grad_output)
    o1 = o1.contiguous()
    grad_qkv1 = [g.contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref, 1, causal)
    g_ref = [get_chunk(g, 1, causal) for g in g_ref]

    # DEBUG CODE
    # d = g_ref[0] - grad_qkv1[0]
    # d1, d2 = d.chunk(2, dim=1)
    # for i in range(get_world_size()):
    #     if i == get_rank():
    #         print(f"rank: {get_rank()}", "block1 diff",d1.max(), "  \n\tblock2 diff", d2.max())
    #     synchronize()
    torch.cuda.synchronize()
    test_msg(check_helper, "Output Correctness Check", o_ref, o1)
    test_msg(check_helper, "Value Correctness Check", g_ref[2], grad_qkv1[2])
    test_msg(check_helper, "Key Correctness Check", g_ref[1], grad_qkv1[1])
    test_msg(check_helper, "Query Correctness Check", g_ref[0], grad_qkv1[0])


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(torch.distributed.get_rank())
    # bmt.init_distributed()
    # bmt.config['sp_stream'] = torch.cuda.Stream(-1)
    for causal in [True, False]:
        for optimize_bwd_comm in [True, False]:
            for deterministic in [True]:
                test_burst(causal, optimize_bwd_comm, deterministic)
    # test_ring_comm()
