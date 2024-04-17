import torch
import bmtrain as bmt
import torch.distributed as dist
from bmtrain.distributed.ops import ncclSend, ncclRecv
from bmtrain.nccl import commCount, groupEnd, groupStart, allReduce

comm_config = {}


def _ring(tensor):
    """Sync send-recv interface"""
    return ring_send_recv(tensor, bmt.rank(), comm_config["comm"])


def ring_send_recv(tensor, rank, comm):
    count = comm_count(comm)
    next_rank = (rank + 1) % count
    prev_rank = (rank - 1 + count) % count
    res = torch.empty_like(tensor, device="cuda", dtype=tensor.dtype)
    global comm_config
    comm_backend = comm_config["backend"]
    if comm_backend == "bmt":
        groupStart()
        ncclSend(tensor.storage(), next_rank, comm)
        ncclRecv(res.storage(), prev_rank, comm)
        groupEnd()
    elif comm_backend == "torch":
        send_op = dist.P2POp(dist.isend, tensor, next_rank, group=None)
        recv_op = dist.P2POp(dist.irecv, res, prev_rank, group=None)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
    return res


def all_reduce(t, c=None):
    if "backend" in comm_config and comm_config["backend"] == "torch":
        t = dist.all_reduce(t, op=dist.ReduceOp.SUM, group=None)
    else:
        if not c:
            c = comm_config["comm"]
        allReduce(t.storage(), t.storage(), "sum", c)


def comm_count(c):
    if "backend" in comm_config and comm_config["backend"] == "torch":
        return dist.get_world_size()
    else:
        return commCount(c)


class ops_wrapper:
    def __init__(self, op, tensor, *args, **kwargs):
        self.op = op
        self.tensor = tensor
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.op(self.tensor.storage(), *self.args, **self.kwargs)


class Ring:
    def __init__(self, comm, rank, backend="torch"):
        self.comm = comm
        self.rank = rank
        self.backend = backend
        self.reqs = []
        self.ops = []

    def ring_send_recv(self, *tensor_list):
        comm = self.comm
        rank = self.rank
        count = comm_count(comm)
        next_rank = (rank + 1) % count
        prev_rank = (rank - 1 + count) % count
        output = []
        i = 0
        for tensor in tensor_list:
            i += 1
            res = torch.zeros_like(tensor)
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

    def commit(self):
        if self.backend == "torch":
            reqs = dist.batch_isend_irecv(self.ops)
        else:
            torch.cuda.synchronize()
            with torch.cuda.stream(comm_config["sp_stream"]):
                for op in self.ops:
                    op.tensor.record_stream(comm_config["sp_stream"])
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
            torch.cuda.current_stream().wait_stream(comm_config["sp_stream"])
        self.reqs = []
        self.ops = []


def print_rank(*args, **kwargs):
    comm_backend = comm_config["backend"]
    if comm_backend == "bmt":
        bmt.print_rank(*args, **kwargs)
    elif comm_backend == "torch":

        def torch_print_rank(*args, **kwargs):
            if dist.get_rank() == 0:
                print(*args, **kwargs)

        torch_print_rank(*args, **kwargs)
    else:
        raise ValueError("Init comm config first")


def synchronize():
    comm_backend = comm_config["backend"]
    if comm_backend == "bmt":
        bmt.synchronize()
    elif comm_backend == "torch":
        dist.barrier()
    else:
        raise ValueError("Init comm config first")


def init_comm_config(backend="bmt"):
    assert backend in [
        "bmt",
        "torch",
    ], "Invalid backend, backend can use `bmt` or `torch`"
    global comm_config

    if backend == "bmt":
        bmt.init_distributed()
        keys = ["comm", "rank", "world_size"]
        for k in keys:
            comm_config[k] = bmt.config[k]
        comm_config["backend"] = "bmt"
    else:
        comm_config["comm"] = None
        torch.distributed.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        comm_config["world_size"] = world_size
        comm_config["rank"] = dist.get_rank()
        comm_config["backend"] = "torch"

    comm_config["sp_stream"] = torch.cuda.Stream(-1)
