import bmtrain as bmt
from burst_attn.comm import ring_bmt,all_reduce
import torch

def test_send_recv():
    bmt.print_rank("Ring Send Recv")
    for i in range(2,10):
        lens = 2 ** i * bmt.world_size()
        part_lens = 2**i
        tensor = torch.randn((lens, 1024, 1024),dtype=torch.float16,device="cuda")
        part_tensor = torch.randn((part_lens, 1024, 1024),dtype=torch.float16,device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for t in range(10):
            for j in range(bmt.world_size()):
                ring_bmt(part_tensor) 
                ring_bmt(part_tensor) 
        start.record()
        for t in range(100):
            for j in range(bmt.world_size()):
                ring_bmt(part_tensor) 
                ring_bmt(part_tensor) 
        end.record()
        torch.cuda.synchronize()
        bmt.print_rank("lens :{} time: {}".format(lens,start.elapsed_time(end)))
def test():
    bmt.print_rank("All Reduce")
    for i in range(2,10):
        lens = 2 ** i * bmt.world_size()
        tensor = torch.randn((lens, 1024, 1024),dtype=torch.float16,device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for t in range(10):
            all_reduce(tensor)
        start.record()
        for t in range(100):
            all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()
        bmt.print_rank("lens :{} time: {}".format(lens,start.elapsed_time(end)))
if __name__ == "__main__":
    bmt.init_distributed()
    test_send_recv()
    test()