import bmtrain as bmt
import torch
from bmtrain.distributed.ops import ncclSend, ncclRecv
from bmtrain.nccl import commCount, groupEnd, groupStart,allReduce

def ring_bmt(tensor):
    return ring_send_recv(tensor, bmt.rank(), bmt.config["comm"])

def all_reduce(tensor):
    comm = bmt.config["comm"]
    tensor = tensor.contiguous()
    allReduce(tensor.storage(),tensor.storage(),"sum",comm)

def ring_send_recv(tensor, rank, comm):

    tensor = tensor.contiguous()
    count = commCount(comm)
    next_rank = (rank + 1) % count
    prev_rank = (rank - 1 + count) % count
    res = torch.ones_like(tensor, device="cuda", dtype=tensor.dtype)
    groupStart()
    if rank%2 == 0:
        ncclSend(tensor.storage(), next_rank, comm)
        ncclRecv(res.storage(), prev_rank, comm)
    else:
        ncclRecv(res.storage(), prev_rank, comm)
        ncclSend(tensor.storage(), next_rank, comm)
    groupEnd()

    return res
