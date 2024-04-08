from benchmark import flash, burst, ref_attn, init_bmt
import torch
import bmtrain as bmt
from checker import check_helper

def get_chunk(t,dim):
    return t.chunk(bmt.world_size(),dim=dim)[bmt.rank()].contiguous()

def test(q,k,v, func,grad_output):
    o = func(q,k,v)
    gq,gk,gv = torch.autograd.grad(o, (q,k,v), grad_output)
    return o,(gq,gk,gv)

def test_burst():
    init_bmt()
    b,s,n,d = 2,1024,16,32
    qkv1 = [torch.randn((b,n,s,d),dtype=torch.float16,device="cuda",requires_grad=True) for _ in range(3)]
    grad_output = torch.randn((b,n,s,d),dtype=torch.float16,device="cuda")
    o_ref,g_ref = test(qkv1[0],qkv1[1],qkv1[2],ref_attn,grad_output) 
    for i in range(3):
        qkv1[i] = qkv1[i].chunk(bmt.world_size(), dim=2)[bmt.rank()]
        qkv1[i] = qkv1[i].transpose(1,2).clone().detach().requires_grad_()
    grad_output = grad_output.transpose(1,2).chunk(bmt.world_size(),dim=1)[bmt.rank()].clone().detach()
    o1, grad_qkv1 = test(qkv1[0],qkv1[1],qkv1[2],burst,grad_output)
    o1 = o1.transpose(1,2)
    grad_qkv1 = [g.transpose(1,2).contiguous() for g in grad_qkv1]
    o_ref = get_chunk(o_ref,dim=2)
    g_ref = [get_chunk(g,dim=2) for g in g_ref]
    check_helper(grad_qkv1[2],g_ref[2])
    check_helper(grad_qkv1[1],g_ref[1])
    check_helper(grad_qkv1[0],g_ref[0])
    if bmt.rank() == 0:
        from IPython import embed;embed()
    bmt.synchronize()
    # check_helper(grad_qkv1[2],g_ref[2])

if __name__ == "__main__":
    
	test_burst()
    
