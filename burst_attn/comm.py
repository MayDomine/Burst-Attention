import torch
import bmtrain as bmt
from bmtrain.distributed.ops import ncclSend, ncclRecv
from bmtrain.nccl import commCount, groupEnd, groupStart,allReduce

def ring_bmt(tensor):
    return ring_send_recv(tensor, bmt.rank(), bmt.config["comm"])

def all_reduce(tensor):
    comm = bmt.config["comm"]
    allReduce(tensor.storage(),tensor.storage(),"sum",comm)

def ring_send_recv(tensor, rank, comm):
    count = commCount(comm)
    next_rank = (rank + 1) % count
    prev_rank = (rank - 1 + count) % count
    res = torch.empty_like(tensor, device="cuda", dtype=tensor.dtype)
    groupStart()
    ncclSend(tensor.storage(), next_rank, comm)
    ncclRecv(res.storage(), prev_rank, comm)
    groupEnd()
    return res
