import torch
import bmtrain as bmt
from bmtrain.distributed import send_activations, recv_activations, reduce_scatter, broadcast, all_gather
import subprocess
from burst_attn.flash_origin import FlashAttnFunc
from burst_attn.burst_attn_simple import OpBurstAttn
from burst_attn.test_ring_attn import ring_attn
from burst_attn.cuda_info import getMemoryTotal

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


def test_multi_gpu(func_name, batch_size, hidden_size, seqlen, num_heads, func, desc, backward=False):
    bmt.init_distributed()
    if func_name == "normal"or func_name == "flash":
        q = torch.randn((batch_size, num_heads, seqlen, hidden_size),device="cuda",dtype=torch.float16).requires_grad_()
        k = torch.randn((batch_size, num_heads, seqlen, hidden_size),device="cuda",dtype=torch.float16).requires_grad_()
        v = torch.randn((batch_size, num_heads, seqlen, hidden_size),device="cuda",dtype=torch.float16).requires_grad_()
    else:
        sub_seq = seqlen // bmt.world_size()
        q = torch.randn((batch_size, num_heads, sub_seq, hidden_size),device="cuda",dtype=torch.float16).requires_grad_()
        k = torch.randn((batch_size, num_heads, sub_seq, hidden_size),device="cuda",dtype=torch.float16).requires_grad_()
        v = torch.randn((batch_size, num_heads, sub_seq, hidden_size),device="cuda",dtype=torch.float16).requires_grad_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        func(q, k, v, backward)
    # output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader']).decode("utf-8").split("\n")[0]
    mem = getMemoryTotal()
    end.record()
    torch.cuda.synchronize()
    if bmt.rank() == 0:
        print(f"{desc} forward: {start.elapsed_time(end)} ms")
        print(f"Memory used:{mem} MiB")

def ref_attn(q, k, v, flash=False):
    scale = q.shape[-1] ** -0.5
    if not flash:
        s = q @ k.transpose(-2, -1) * scale
        s = torch.softmax(s, dim=-1)
        p = s @ v
    else:
        from flash_attn.flash_attn_triton import FlashAttnFunc as flash_func
        batch_size,_,seqlen,_ = q.shape
        q = q.transpose(1,2).contiguous()
        k = k.transpose(1,2).contiguous()
        v = v.transpose(1,2).contiguous()
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=q.device)
        # func = lambda q,k,v,bias,causal,sm_scale:flash_func.apply(q,k,v,cu_seqlens,cu_seqlens,seqlen,seqlen,0,sm_scale,causal,False,False)
        p = FlashAttnFunc.apply(q, k ,v ,None, False, scale).transpose(1,2)
        # p = func(q,k,v,None,False,scale)
    return p

def test_ref(q, k ,v, backward=False, flash=False):
    res_ref = ref_attn(q, k, v, flash)
    g = torch.randn_like(res_ref)
    if backward:
        torch.autograd.grad(res_ref, (q, k, v), g)

def test_burst(q, k, v, backward=False, flash=False):
    # sub_seq = q.shape[2] // bmt.world_size()
    # q = q[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    # k = k[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    # v = v[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    res_burst = OpBurstAttn.apply(q, k, v, None, flash)
    if backward:
        g = torch.ones_like(res_burst)
        dq,dk,dv=torch.autograd.grad(res_burst, (q, k, v), g)
        return dq
    

def test_ring(q, k ,v, backward=False):
    # sub_seq = q.shape[2] // bmt.world_size()
    # q = q[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    # k = k[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    # v = v[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    res_ring = ring_attn(q,k,v)
    if backward:
        g = torch.ones_like(res_ring)
        dq,dk,dv=torch.autograd.grad(res_ring, (q, k, v), g)
        return dq

def test_backward():
    batch = 2
    seqlen = 1024
    head_dim = 32
    num_heads = 16
    bmt.init_distributed()
    q_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda",dtype=torch.float16)
    k_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda",dtype=torch.float16)
    v_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda",dtype=torch.float16)
    q_whole = broadcast(q_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    k_whole = broadcast(k_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    v_whole = broadcast(v_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    res1 = test_burst(q_whole,k_whole,v_whole,True)
    res2 = test_ring(q_whole,k_whole,v_whole,True)
    if bmt.rank() == 1:
        print(res1[0])
        print(res2[0])

def test():
    batch = 2
    seqlen = 1024
    head_dim = 32
    num_heads = 16
    bmt.init_distributed()
    q_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda",dtype=torch.float16)
    k_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda",dtype=torch.float16)
    v_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda",dtype=torch.float16)
    q_whole = broadcast(q_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    k_whole = broadcast(k_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    v_whole = broadcast(v_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    sub_seq = seqlen // bmt.world_size()
    q = q_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_().contiguous()
    k = k_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_().contiguous()
    v = v_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_().contiguous()
    res_burst = OpBurstAttn.apply(q, k, v, None, True)
    # res = rearrange(res_burst, "b n s h -> s n b h")
    # res = all_gather(res, bmt.config["comm"]).flatten(0,1)
    # res_whole = rearrange(res, "s n b h -> b n s h")
    res2 = ref_attn(q_whole, k_whole, v_whole, flash=False)
    g = torch.randn_like(res2)
    dq_ref,dk_ref,dv_ref = torch.autograd.grad(res2, (q_whole, k_whole, v_whole), g)
    g2 = g[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    print(g2.shape)

    dq,dk,dv = torch.autograd.grad(res_burst, (q, k, v), g2)
    dq_ref_sub = dq_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dk_ref_sub = dk_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dv_ref_sub = dv_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    # print(torch.allclose(dq, dq_ref_sub, atol=1e-4))
    # print(torch.allclose(dk, dk_ref_sub, atol=1e-4))
    # print(torch.allclose(dv, dv_ref_sub, atol=1e-4))
    if bmt.rank() == 0:
        print((dq-dq_ref_sub).abs().max())
        print((dk-dk_ref_sub).abs().max())
        print((dv-dv_ref_sub).abs().max())



if __name__ == "__main__":
    # test_backward()
    test()
    # import argparse
    # parser = argparse.ArgumentParser(description='Test multi-gpu function')

    # # 添加命令行参数
    # parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    # parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size')
    # parser.add_argument('--seqlen', type=int, default=128, help='Sequence length')
    # parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    # parser.add_argument('--func', type=str, default='sigmoid', help='Activation function')
    # parser.add_argument('--desc', type=str, default="Test", help='Description')
    # parser.add_argument("--backward",action="store_true")
    # # 解析命令行参数
    # args = parser.parse_args()
    # batch_size = args.batch_size
    # hidden_size = args.hidden_size
    # seqlen = args.seqlen
    # num_heads = args.num_heads
    # if args.func == "burst":
    #     func = test_burst
    # elif args.func == "normal":
    #     func = test_ref
    # elif args.func == 'ring':
    #     func = test_ring
    # elif args.func == "flash":
    #     func = lambda q,k,v,backward: test_ref(q,k,v,backward,flash=True)
    # elif args.func == "burst_flash":
    #     func = lambda q,k,v,backward: test_burst(q,k,v,backward,flash=True)
    
    # test_multi_gpu(args.func, batch_size, hidden_size, seqlen,num_heads, func, args.desc, args.backward)