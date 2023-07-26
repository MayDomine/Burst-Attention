import torch
import bmtrain as bmt
from layers import Linear

class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None, act = "relu") -> None:
        super().__init__()

        self.w_in = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
        self.w_out = Linear(dim_ff, dim_model, bias = bias, dtype=dtype)
        if act.startswith("gated"):
            self.gated = True
            act = act.split("_")[-1]
        else:
            self.gated = False
        if self.gated:
            self.w_gated = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
        act_map = {"relu":torch.nn.ReLU,"silu":torch.nn.functional.silu}
        self.act = act_map[act]()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        if self.gated:
            gate_score = self.relu(self.w_gated(input))
            output = self.w_in(input) * gate_score
        else:
            output = self.act(self.w_in(input))
        return self.w_out(output)
