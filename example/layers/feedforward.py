import torch
import bmtrain as bmt
from layers import Linear
act = {
    "relu":torch.nn.ReLU(),
    "gelu":torch.nn.GELU(),
    "silu":torch.nn.functional.silu, 
}
class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None, gated=False) -> None:
        super().__init__()
        self.gated = gated
        if bmt.config['tp_size'] > 1:
            if gated:
                self.w_in_1 = bmt.nn.ColumnParallelLinear(dim_model, dim_ff, bias = bias, dtype=dtype,gather_input=False,async_gather_chunks=1)
                self.w_in_2 = bmt.nn.ColumnParallelLinear(dim_model, dim_ff, bias = bias, dtype=dtype,gather_input=False,async_gather_chunks=1)
            else:
                self.w_in = bmt.nn.ColumnParallelLinear(dim_model, dim_ff, bias = bias, dtype=dtype,gather_input=False,async_gather_chunks=1)
            self.w_out = bmt.nn.RowParallelLinear(dim_ff, dim_model, bias = bias, dtype=dtype,all_reduce_output=True,async_chunks=1)
        else:
            if gated:
                self.w_in = Linear(dim_model, dim_ff*2, bias = bias, dtype=dtype)
            else:
                self.w_in = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
            self.w_out = Linear(dim_ff, dim_model, bias = bias, dtype=dtype)
        self.act = act[bmt.config["act"]]
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        if self.gated:
            if bmt.config["tp_size"] > 1:
                gated = self.w_in_1(input)
                output = self.w_in_2(input)
            else:
                gated, output = self.w_in(input).chunk(2,dim=-1)
            gated = self.act(gated)
            output = gated * output
        else:
            output = self.act(self.w_in(input)) 
        return self.w_out(output)
