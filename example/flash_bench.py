import bmtrain as bmt
from burst_attn.comm import ring_bmt,all_reduce
import torch
import math
def flash(h_q,h_k,h_v):
    from flash_attn.flash_attn_triton import FlashAttnFunc as flash_func 
    dim_head = h_q.shape[-1]
    func = lambda q,k,v,bias,causal,sm_scale:flash_func.apply(q,k,v,bias, causal, sm_scale)
    h_out = func(h_q, h_k ,h_v ,None,False,1/math.sqrt(dim_head))
    return h_out
def test_flash_seqlen():
    bmt.print_rank("split seqlen")
    for i in range(2,15):
        lens = 2 ** i * bmt.world_size()
        part_lens = 2**i
        num_head = 64
        dim_head = 128
        part_tensor = torch.randn((1, part_lens, num_head, dim_head),dtype=torch.float16,device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for t in range(10):
            flash(part_tensor,part_tensor,part_tensor)
        start.record()
        for t in range(100):
            flash(part_tensor,part_tensor,part_tensor)
        end.record()
        torch.cuda.synchronize()
        bmt.print_rank("lens :{} time: {}".format(lens,start.elapsed_time(end)))
def test():
    bmt.print_rank("split num_head")
    for i in range(2,15):
        lens = 2 ** i * bmt.world_size()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        num_head = 64 // bmt.world_size()
        dim_head = 128
        part_tensor = torch.randn((1, lens, num_head, dim_head),dtype=torch.float16,device="cuda")
        for t in range(10):
            flash(part_tensor,part_tensor,part_tensor)
        start.record()
        for t in range(100):
            flash(part_tensor,part_tensor,part_tensor)
        end.record()
        torch.cuda.synchronize()
        bmt.print_rank("lens :{} time: {}".format(lens,start.elapsed_time(end)))
if __name__ == "__main__":
    bmt.init_distributed()
    test_flash_seqlen()
    test()