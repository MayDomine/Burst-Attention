import sys
import torch
import os
import bmtrain as bmt
import numpy as np
import argparse

sys.path.append("..")
from flash_attn.flash_attn_interface import flash_attn_func as flash_cuda
from burst_attn.comm import synchronize, get_rank, get_world_size
from checker import check_helper
from burst_attn import OpBurstAttn
from burst_attn.comm import Ring, get_rank, get_world_size, print_rank, gather_obj, broadcast
from burst_attn.log_helper import get_logger
_logger = get_logger(__file__, level="DEBUG")

def test_msg(test_func, msg, *args, **kwargs):
    try:
        e = test_func(*args, **kwargs)
        success = 1
    except Exception as e:
        success = 0
    res = gather_obj(success)
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


def burst_func(q, k, v, causal=False, optimize_bwd_comm=False, deterministic=False, group=None, sub_group=[None,None]):
    return OpBurstAttn.apply(
        q, k, v, None, "cuda", causal, optimize_bwd_comm, deterministic, group, sub_group
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


def get_group(create_dq_group=True):
    # if dq use the same group with q, g, lse, softmax_d, it wont pass the correctness pass in double_ring
    if bmt.init.is_initialized():
        return (
            bmt.config["comm"],
            (bmt.config["local_comm"], bmt.config["local_comm2"]),
            (bmt.config["local_idx_comm"], bmt.config["local_idx_comm2"])
        )
    else:
        local_size = get_world_size() // 4
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
            return None, (intra_group, intra_group2,), (inter_group, inter_group2)


def test_burst(
    causal=False,
    optimize_bwd_comm=False,
    deterministic=True,
    double_ring=True,
    group=None,
):
    print_rank(
        f"Checking Burst Attn Causal = {causal}... Optimize Bwd Comm = {optimize_bwd_comm}... Deterministic = {deterministic}... Double Ring = {double_ring}..."
    )
    b, s, n, d = 2, 2048, 32, 128
    qkv = torch.randn(b, s * 3, n, d, dtype=torch.float16).cuda()
    grad_output = torch.randn(b, s, n, d, dtype=torch.float16).cuda()
    synchronize()
    flash = lambda q, k, v: flash_cuda(q, k, v, causal=causal, softmax_scale=None)
    qkv = broadcast(qkv, 0)
    grad_output = broadcast(grad_output, 0)
    synchronize()
    qkv1 = [t.clone().detach().requires_grad_() for t in qkv.chunk(3, dim=1)]
    qkv1_buf = [None] * 3
    for i in range(3):
        qkv1_buf[i] = get_chunk(qkv1[i], 1, causal).detach().clone().requires_grad_()

    o_ref, g_ref = test(qkv1[0], qkv1[1], qkv1[2], flash, grad_output)
    grad_output = get_chunk(grad_output, 1, causal)
    grad_output = grad_output.clone().detach().contiguous()
    for i in range(3):
        qkv1[i] = qkv1_buf[i]
    qkv1 = [t.contiguous() for t in qkv1]
    if double_ring:
        group, intra_g, inter_g = get_group()
    else:
        group, intra_g, inter_g = None, None, None
    o1 = burst_func(
        qkv1[0],
        qkv1[1],
        qkv1[2],
        causal,
        optimize_bwd_comm,
        deterministic,
        group,
        [intra_g, inter_g]
    )
    grad_qkv1 = torch.autograd.grad(o1, qkv1, grad_output)
    # o1, grad_qkv1 = test(qkv1[0], qkv1[1], qkv1[2], burst_func, grad_output)
    o1 = o1.contiguous()
    grad_qkv1 = [g.contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref, 1, causal)
    g_ref = [get_chunk(g, 1, causal) for g in g_ref]

    torch.cuda.synchronize()
    synchronize()
    test_msg(check_helper, "Output Correctness Check", o_ref, o1)
    test_msg(check_helper, "Value Correctness Check", g_ref[2], grad_qkv1[2])
    test_msg(check_helper, "Key Correctness Check", g_ref[1], grad_qkv1[1])
    test_msg(check_helper, "Query Correctness Check", g_ref[0], grad_qkv1[0])
    return 1

def make_cmd():
    for causal in [False, True]:
        for optimize_bwd_comm in [True, False]:
            for deterministic in [False, True]:
                for double_ring in [True]:
                    cmd = "torchrun --nnodes 1 --nproc_per_node 8 test_burst.py"
                    if causal:
                        cmd += " --causal"
                    if optimize_bwd_comm:
                        cmd += " --optimize_bwd_comm"
                    if deterministic:
                        cmd += " --deterministic"
                    if double_ring:
                        cmd += " --double_ring"
                    print(cmd)
def test_all():
    for causal in [False, True]:
        for optimize_bwd_comm in [True, False]:
            for deterministic in [False, True]:
                for double_ring in [True, False]:
                    success = test_burst(causal, optimize_bwd_comm, deterministic, double_ring)

def bmt_init():
    bmt.init_distributed()
    bmt.config['sp_stream'] = torch.cuda.Stream(-1)
    bmt.config['sp_stream2'] = torch.cuda.Stream(-1)
    bmt.config['sp_stream3'] = torch.cuda.Stream(-1)
    bmt.config['sp_stream4'] = torch.cuda.Stream(-1)


def test_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--optimize_bwd_comm", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--double_ring", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    # torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # torch.cuda.set_device(torch.distributed.get_rank())
    bmt_init()
    if args.all:
        test_all()
    else:
        test_burst(
            args.causal,
            args.optimize_bwd_comm,
            args.deterministic,
            args.double_ring,
        )
if __name__ == "__main__":
    test_cli()
