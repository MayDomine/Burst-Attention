import bmtrain as bmt
import torch
from bmtrain.distributed import send_activations, recv_activations, reduce_scatter, broadcast, all_gather
from einops import rearrange, repeat
from .comm import ring_bmt
# from .test_ring_attn import ring_attn
# from flash_attn.flash_attn_triton2 import 
from .flash_attn import _flash_attn_forward,_flash_attn_backward
import subprocess

def inter_normal_attn(q, k, v, m_i, acc_o, softmax_scale=1.0):
    m_i = m_i.to(q.dtype) if m_i is not None else None
    qk = q @ k.transpose(-2, -1)*softmax_scale
   
    m_ij = torch.max(qk, dim=-1, keepdim=True)[0]
    if m_i is not None:
        m_ij = torch.maximum(m_ij, m_i)
    p = torch.exp(qk - m_ij)
    l_ij = torch.sum(p, dim=-1, keepdim=True)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i-m_ij)
        acc_o = p @ v + acc_o_scale * acc_o
    else:
        acc_o = p @ v
    return acc_o,m_ij,l_ij

def inter_normal_attn_backward(do, q, k, v, delta, lse, d_q, d_k, d_v, softmax_scale):
    qk = q @ k.transpose(-2, -1) * softmax_scale
    p = torch.exp(qk - lse) 
    d_v += p.transpose(-2, -1) @ do
    d_p = do @ v.transpose(-2, -1)
    softmax_scale = softmax_scale
    d_s = p * (d_p - delta) * softmax_scale
    d_q += d_s @ k
    d_k += d_s.transpose(-2, -1) @ q

def inter_flash_attn(q, k, v, m_i, acc_o, softmax_scale=1.0):
    q = q.transpose(1,2).contiguous()
    k = k.transpose(1,2).contiguous()
    v = v.transpose(1,2).contiguous()
    m_i = m_i.squeeze().contiguous() if m_i is not None else None
    if m_i is None:
        b,s,n,d = q.shape
        m_i = (-torch.ones((b,n,s),dtype=torch.float32,device="cuda") * torch.inf).contiguous()
    o,l_ij,m_ij,softamx_scale = _flash_attn_forward(q,k,v,m_i, causal=False,softmax_scale=softmax_scale)
    o = o.transpose(1,2).contiguous()
    if m_i is not None:
        m_ij = torch.maximum(m_i, m_ij)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i-m_ij).to(q.dtype)
        acc_o_scale = acc_o_scale.unsqueeze(-1).contiguous()
        acc_o = o + acc_o * acc_o_scale
    else:
        acc_o = o
    l_ij = torch.exp(l_ij - m_ij)
    m_ij = m_ij.unsqueeze(-1).contiguous()
    l_ij = l_ij.unsqueeze(-1).contiguous()
    return acc_o,m_ij,l_ij

def inter_flash_attn_backward(do,q,k,v,delta,lse,dq,dk,dv,softmax_scale):
    lse =  lse.squeeze(-1).contiguous()
    delta = delta.squeeze(-1).contiguous()
    do = do.transpose(1,2).contiguous()
    q = q.transpose(1,2).contiguous() #B N S D -> B S N H
    k = k.transpose(1,2).contiguous() 
    v = v.transpose(1,2).contiguous()
    dq_ = torch.empty_like(q)
    dk_ = torch.empty_like(k)
    dv_ = torch.empty_like(v)
    _flash_attn_backward(do, q, k, v, delta, lse, dq_, dk_, dv_, softmax_scale=softmax_scale)
    dq += dq_.transpose(1,2).contiguous()
    dk += dk_.transpose(1,2).contiguous()
    dv += dv_.transpose(1,2).contiguous()


class OpBurstAttn(torch.autograd.Function):
    """
    q, k, v: [B, N, S, H] (batch_size, num_heads, sub_seqlen, head_dim)
    bias: [N, S, S] (num_heads, sub_seqlen, sub_seqlen)
    """
    @staticmethod
    def forward(ctx, query, key ,value, mask, softmax_scale=1.0, flash=False):
        
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        
        ctx.softmax_scale = softmax_scale
        m_i = None
        acc_o = None
        ctx.flash=flash
        if ctx.flash:
            forward_func = inter_flash_attn
        else:
            forward_func = inter_normal_attn
        acc_o,m_i,l_ij = forward_func(q, k, v, m_i, acc_o, softmax_scale)
        lse_i = torch.log(l_ij) + m_i
        for j in range(bmt.world_size()-1):
            k = ring_bmt(k)
            v = ring_bmt(v)
            if not ctx.flash:
                with torch.no_grad():
                    acc_o,m_ij,l_ij = forward_func(q, k, v, m_i, acc_o, softmax_scale)
            else:
                acc_o,m_ij,l_ij = forward_func(q, k, v, m_i, acc_o, softmax_scale)
            m_i = m_ij
            l_i_new = torch.exp(lse_i - m_ij) + l_ij
            lse_i = torch.log(l_i_new) + m_ij
        o_scale = torch.exp(m_i - lse_i).to(q.dtype)
        acc_o = acc_o * o_scale
        ctx.save_for_backward(query, key, value, lse_i, acc_o)
        return acc_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, lse_i, o_i = ctx.saved_tensors
        batch_size, num_heads, seqlen, head_dim = q.shape
        d_q_whole = torch.zeros(batch_size, num_heads, seqlen*bmt.world_size(), head_dim, device=q.device)
        d_q = torch.zeros_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        delta = (o_i * grad_output).sum(-1, keepdim=True)
        if ctx.flash:
            backward_func = inter_flash_attn_backward
        else:
            backward_func = inter_normal_attn_backward
        start = bmt.rank() * seqlen
        end = (bmt.rank() + 1) * seqlen
        backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:end, :], d_k, d_v, ctx.softmax_scale)
        for j in range(bmt.world_size()-1):
            delta = ring_bmt(delta)
            q = ring_bmt(q)
            grad_output = ring_bmt(grad_output)
            lse_i = ring_bmt(lse_i)
            start = (start - seqlen) % (seqlen * bmt.world_size())
            end = (end - seqlen) % (seqlen * bmt.world_size())
            if end < start:
                backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:, :], d_k, d_v, ctx.softmax_scale)
            else:
                backward_func(grad_output, q, k, v, delta, lse_i, d_q_whole[:, :, start:end, :], d_k, d_v, ctx.softmax_scale)
        d_q_whole = rearrange(d_q_whole, "b n s h -> s n b h")
        d_q = reduce_scatter(d_q_whole, "sum",bmt.config["comm"])
        d_q = rearrange(d_q, "s n b h -> b n s h")
        return d_q, d_k, d_v, None, None, None
    
def test_multi_gpu(batch_size,hidden_size,seqlen,num_heads,func,desc,backward=False):
    bmt.init_distributed()
    q_whole = torch.randn((batch_size, num_heads, seqlen, hidden_size),device="cuda",dtype=torch.float32)
    k_whole = torch.randn((batch_size, num_heads, seqlen, hidden_size),device="cuda",dtype=torch.float32)
    v_whole = torch.randn((batch_size, num_heads, seqlen, hidden_size),device="cuda",dtype=torch.float32)
    q_whole = broadcast(q_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    k_whole = broadcast(k_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    v_whole = broadcast(v_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1):
        func(q_whole,k_whole,v_whole,backward)
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader']).decode("utf-8").split("\n")[0]
    end.record()
    torch.cuda.synchronize()
    if bmt.rank() == 0:
        print(f"{desc} forward: {start.elapsed_time(end)} ms")
        print(f"Memory used:{output}")

def ref_attn(q, k, v):
    s = q @ k.transpose(-2, -1)
    s = torch.softmax(s, dim=-1)
    p = s @ v
    return p

def test_ref(q, k ,v, backward=False):
    res_ref = ref_attn(q, k, v)
    g = torch.randn_like(res_ref)
    if backward:
        torch.autograd.grad(res_ref, (q, k, v), g)

def test_burst(q, k, v, backward=False):
    sub_seq = q.shape[2] // bmt.world_size()
    q = q[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    k = k[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    v = v[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    res_burst = OpBurstAttn.apply(q, k, v, None)
    if backward:
        g = torch.ones_like(res_burst)
        dq,dk,dv=torch.autograd.grad(res_burst, (q, k, v), g)
    return dq
def test_ring(q, k ,v, backward=False):
    sub_seq = q.shape[2] // bmt.world_size()
    q = q[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    k = k[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    v = v[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    res_ring = ring_attn(q,k,v)
    if backward:
        g = torch.ones_like(res_ring)
        dq,dk,dv=torch.autograd.grad(res_ring, (q, k, v), g)
    return dq
# def test_falsh(q,k,v,backward=False):
def test_backward():
    batch = 2
    seqlen = 1024
    head_dim = 32
    num_heads = 16
    bmt.init_distributed()
    q_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    k_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    v_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
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
    q_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    k_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    v_whole = torch.randn((batch, num_heads, seqlen, head_dim),device="cuda")
    q_whole = broadcast(q_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    k_whole = broadcast(k_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    v_whole = broadcast(v_whole, 0, bmt.config["comm"]).detach().requires_grad_()
    sub_seq = seqlen // bmt.world_size()
    q = q_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    k = k_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    v = v_whole[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :].detach().requires_grad_()
    res_burst = OpBurstAttn.apply(q, k, v, None)
    # res = rearrange(res_burst, "b n s h -> s n b h")
    # res = all_gather(res, bmt.config["comm"]).flatten(0,1)
    # res_whole = rearrange(res, "s n b h -> b n s h")
    res2 = ref_attn(q_whole, k_whole, v_whole)
    g = torch.randn_like(res2)
    dq_ref,dk_ref,dv_ref = torch.autograd.grad(res2, (q_whole, k_whole, v_whole), g)
    g2 = g[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dq,dk,dv = torch.autograd.grad(res_burst, (q, k, v), g2)
    dq_ref_sub = dq_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dk_ref_sub = dk_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    dv_ref_sub = dv_ref[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq, :]
    # print(torch.allclose(dq, dq_ref_sub, atol=1e-4))
    # print(torch.allclose(dk, dk_ref_sub, atol=1e-4))
    # print(torch.allclose(dv, dv_ref_sub, atol=1e-4))
    print((dq-dq_ref_sub).abs().max())
    print((dk-dk_ref_sub).abs().max())
    print((dv-dv_ref_sub).abs().max())



if __name__ == "__main__":
    # test_backward()
    # test()
    import argparse
    parser = argparse.ArgumentParser(description='Test multi-gpu function')

    # 添加命令行参数
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size')
    parser.add_argument('--seqlen', type=int, default=128, help='Sequence length')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--func', type=str, default='sigmoid', help='Activation function')
    parser.add_argument('--desc', type=str, default="Test", help='Description')
    parser.add_argument("--backward",action="store_true")
    # 解析命令行参数
    args = parser.parse_args()
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    seqlen = args.seqlen
    num_heads = args.num_heads
    if args.func == "burst":
        func = test_burst
    elif args.func == "normal":
        func = test_ref
    elif args.func == 'ring':
        func = test_ring
    
    test_multi_gpu(batch_size,hidden_size,seqlen,num_heads,func,args.desc,args.backward)