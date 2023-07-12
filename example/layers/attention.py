from typing import Optional
import torch
import bmtrain as bmt
from layers import Linear
import math
from burst_attn.burst_attn_simple import OpBurstAttn
from burst_attn.flash_origin import FlashAttnFunc
from burst_attn.test_ring_attn import ring_attn
from einops import rearrange
class Attention(bmt.DistributedModule):
    def __init__(self, 
            dim_model : int, dim_head : int,
            num_heads : int, bias : bool = True,
            dtype = None,
            sequence_parallel : bool = False,
            sequence_parallel_impl: str = "burst",
            flash: bool = False,
        ) -> None:
        super().__init__()

        self.project_q = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_k = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_v = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.sequence_parallel_impl = sequence_parallel_impl
        self.project_out = Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)
        self.flash = flash
        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model
        self.sequence_parallel = sequence_parallel
    
    def forward(self, 
            hidden_q : torch.Tensor,        # (batch_size, seq_q, dim_model)
            hidden_kv : torch.Tensor,       # (batch_size, seq_kv, dim_model)
            mask : torch.BoolTensor,        # (batch_size, seq_q, seq_kv)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_heads, seq_q, seq_kv)
        ) -> torch.Tensor:
        batch_size, seq_q, dim_model = hidden_q.size()
        seq_kv = hidden_kv.size(1)

        h_q : torch.Tensor = self.project_q(hidden_q)
        h_k : torch.Tensor = self.project_k(hidden_kv)
        h_v : torch.Tensor = self.project_v(hidden_kv)

        h_q = h_q.view(batch_size, seq_q, self.num_heads, self.dim_head)
        h_k = h_k.view(batch_size, seq_kv, self.num_heads, self.dim_head)
        h_v = h_v.view(batch_size, seq_kv, self.num_heads, self.dim_head)
        
        h_q = h_q.permute(0, 2, 1, 3).contiguous()
        h_k = h_k.permute(0, 2, 1, 3).contiguous()
        h_v = h_v.permute(0, 2, 1, 3).contiguous()

        if not self.sequence_parallel:
            if self.flash:
                batch_size,_,seqlen,_ = h_q.shape
                h_q = h_q.permute(0, 2, 1, 3).flatten(0,1).contiguous()
                h_k = h_k.permute(0, 2, 1, 3).flatten(0,1).contiguous()
                h_v = h_v.permute(0, 2, 1, 3).flatten(0,1).contiguous()
                from flash_attn.flash_attn_interface import FlashAttnFunc as flash_func
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=h_q.device)
                func = lambda q,k,v,bias,causal,sm_scale:flash_func.apply(h_q,h_k,h_v,cu_seqlens,cu_seqlens,seqlen,seqlen,0,sm_scale,causal,False,False)
                from flash_attn.flash_attn_interface import FlashAttnFunc as flash_func
                # func = lambda q,k,v,bias,causal,sm_scale:flash_func(q,k,v,q.shape[2],k.shape[2],q.shape[2],k.shape[2],0,sm_scale,causal,False,False)
                h_out = func(h_q, h_k ,h_v ,None,False,1/math.sqrt(self.dim_head))
                h_out = rearrange(h_out,"(b s) n h -> b s n h",b = batch_size)
                h_out = h_out.permute(0, 2, 1, 3).contiguous()
            else:
                h_q = h_q.view(batch_size * self.num_heads, seq_q, self.dim_head)
                h_k = h_k.view(batch_size * self.num_heads, seq_kv, self.dim_head)
                h_v = h_v.view(batch_size * self.num_heads, seq_kv, self.dim_head)
                score = torch.bmm(
                    h_q, h_k.transpose(1, 2)
                )
                score = score / math.sqrt(self.dim_head)

                score = score.view(batch_size, self.num_heads, seq_q, seq_kv)

                if position_bias is not None:
                    score = score + position_bias.view(batch_size, self.num_heads, seq_q, seq_kv)
                
                score = torch.where(
                    mask.view(batch_size, 1, seq_q, seq_kv),
                    score,
                    torch.scalar_tensor(float('-inf'), device=score.device, dtype=score.dtype)
                )

                score = torch.where(
                    mask.view(batch_size, 1, seq_q, seq_kv),
                    self.softmax(score),
                    torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
                )

                score = score.view(batch_size * self.num_heads, seq_q, seq_kv)

                h_out = torch.bmm(
                    score, h_v
                )
                h_out = h_out.view(batch_size, self.num_heads, seq_q, self.dim_head)
            # if bmt.rank() == 0:
            #     print(h_out[0])
        else:
            if self.sequence_parallel_impl == "burst":
                scale = math.sqrt(self.dim_head)
                h_out = OpBurstAttn.apply(h_q,h_k,h_v,1.0/scale, self.flash)
            elif self.sequence_parallel_impl == "ring":
                b = h_q.shape[0]
                scale = math.sqrt(self.dim_head)
                h_out = ring_attn(h_q,h_k,h_v,scale)
                h_out = rearrange(h_out, "(b n) s d -> b n s d", b=b)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        h_out = h_out.view(batch_size, seq_q, self.num_heads * self.dim_head)

        attn_out = self.project_out(h_out)
        return attn_out
        

        


