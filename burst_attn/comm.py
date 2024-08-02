import torch
import bmtrain as bmt
import torch.distributed as dist
from bmtrain.distributed.ops import ncclSend, ncclRecv
from bmtrain.nccl import commCount, groupEnd, groupStart, allReduce, commRank


def is_bmt_enable():
    return bmt.init.is_initialized()


def _ring(tensor):
    """Sync send-recv interface"""
    return ring_send_recv(tensor, bmt.rank(), bmt.config["comm"])


def ring_send_recv(tensor, comm):
    """Sync send-recv interface"""

    rank = get_rank()
    count = get_world_size(comm)
    next_rank = (rank + 1) % count
    prev_rank = (rank - 1 + count) % count
    res = torch.empty_like(tensor, device="cuda", dtype=tensor.dtype)
    if is_bmt_enable():
        groupStart()
        ncclSend(tensor.storage(), next_rank, comm)
        ncclRecv(res.storage(), prev_rank, comm)
        groupEnd()
    else:
        send_op = dist.P2POp(dist.isend, tensor, next_rank, group=comm)
        recv_op = dist.P2POp(dist.irecv, res, prev_rank, group=comm)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
    return res


def all_reduce(t, group=None):
    if not is_bmt_enable():
        t = dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    else:
        group = bmt.config["comm"] if not group else group
        allReduce(t.storage(), t.storage(), "sum", group)


def get_world_size(c=None):
    if not is_bmt_enable():
        return dist.get_world_size(c)
    else:
        c = bmt.config["comm"] if not c else c
        return commCount(c)


class ops_wrapper:
    def __init__(self, op, tensor, *args, **kwargs):
        self.op = op
        self.tensor = tensor
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.op(self.tensor.storage(), *self.args, **self.kwargs)


def get_rank(group=None):
    if is_bmt_enable():
        group = bmt.config["comm"] if not group else group
        return commRank(group)
    else:
        return dist.get_rank(group)


class Ring:
    def __init__(self, process_group, backend="torch"):
        if is_bmt_enable():
            if process_group:
                self.comm = process_group
            else:
                self.comm = bmt.config["comm"]
            self.backend = "bmtrain"
        else:
            self.comm = process_group
            self.backend = "torch"
        self.world_size = get_world_size(process_group)
        self.rank = get_rank(process_group)
        self.reqs = []
        self.ops = []
    def _ring_send_recv_tensor_base(self, src_tensor, dst_tensor):
        # no handle is connected with Ring object 
        comm = self.comm
        rank = self.rank
        count = get_world_size(comm)
        next_rank = (rank + 1) % count
        prev_rank = (rank - 1 + count) % count
        ops = []
        if self.backend == "torch":
            send_op = dist.P2POp(dist.isend, src_tensor, next_rank, group=None)
            recv_op = dist.P2POp(dist.irecv, dst_tensor, prev_rank, group=None)
        else:
            send_op = ops_wrapper(ncclSend, src_tensor, next_rank, comm)
            recv_op = ops_wrapper(ncclRecv, dst_tensor, prev_rank, comm)
        ops.append(send_op)
        ops.append(recv_op)
        if rank % 2 == 0:
            ops.append(send_op)
            ops.append(recv_op)
        else:
            ops.append(recv_op)
            ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(ops)
        return ops

    def ring_send_recv(self, *tensor_list):
        comm = self.comm
        rank = self.rank
        count = get_world_size(comm)
        next_rank = (rank + 1) % count
        prev_rank = (rank - 1 + count) % count
        output = []
        i = 0
        for tensor in tensor_list:
            i += 1
            res = torch.zeros_like(tensor)
            # res  = tensor
            if self.backend == "torch":
                send_op = dist.P2POp(dist.isend, tensor, next_rank, group=None)
                recv_op = dist.P2POp(dist.irecv, res, prev_rank, group=None)
            else:
                send_op = ops_wrapper(ncclSend, tensor, next_rank, comm)
                recv_op = ops_wrapper(ncclRecv, res, prev_rank, comm)
            self.ops.append(send_op)
            self.ops.append(recv_op)
            output.append(res)
        return output

    def _ring_send_recv_base(self, tensor_list, dest_list):
        comm = self.comm
        rank = self.rank
        count = get_world_size(comm)
        next_rank = (rank + 1) % count
        prev_rank = (rank - 1 + count) % count
        i = 0
        for send_t, recv_t in zip(tensor_list, dest_list):
            i += 1
            if self.backend == "torch":
                send_op = dist.P2POp(dist.isend, send_t, next_rank, group=None)
                recv_op = dist.P2POp(dist.irecv, recv_t, prev_rank, group=None)
            else:
                send_op = ops_wrapper(ncclSend, send_t, next_rank, comm)
                recv_op = ops_wrapper(ncclRecv, recv_t, prev_rank, comm)
            self.ops.append(send_op)
            self.ops.append(recv_op)

    def commit(self):
        if self.backend == "torch":
            reqs = dist.batch_isend_irecv(self.ops)
        else:
            torch.cuda.synchronize()
            with torch.cuda.stream(bmt.config["sp_stream"]):
                for op in self.ops:
                    op.tensor.record_stream(bmt.config["sp_stream"])
                groupStart()
                for op in self.ops:
                    op()
                groupEnd()
            reqs = None
        self.reqs = reqs

    def wait(self):
        if self.backend == "torch":
            for req in self.reqs:
                req.wait()
        else:
            torch.cuda.current_stream().wait_stream(bmt.config["sp_stream"])
        self.reqs = []
        self.ops = []


def print_rank(*args, **kwargs):
    if is_bmt_enable():
        bmt.print_rank(*args, **kwargs)
    else:

        def torch_print_rank(*args, **kwargs):
            if dist.get_rank() == 0:
                print(*args, **kwargs)

        torch_print_rank(*args, **kwargs)


def synchronize():
    if is_bmt_enable():
        bmt.synchronize()
    elif dist.is_initialized():
        dist.barrier()
    else:
        raise ValueError("Init comm first")
