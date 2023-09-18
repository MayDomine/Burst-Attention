import torch
import bmtrain as bmt
from layers import TransformerEncoder, Layernorm, Embedding, TransformerEncoder, RotaryEmbeddingESM,Linear

class LLama(bmt.DistributedModule):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None,sequence_parallel : bool = False,flash: bool = False,
            sequence_parallel_impl="burst",act="gated_silu"
        ) -> None:
        super().__init__()

        self.max_distance = max_distance

        self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        self.position_bias = RotaryEmbeddingESM(
            dim = dim_head,
            dtype = dtype,
        )
        
        self.transformers = bmt.TransformerBlockList([
            bmt.ZeROBlock(
                TransformerEncoder(
                    dim_model, dim_head, num_heads, dim_ff, bias, dtype,sequence_parallel,flash,sequence_parallel_impl
                )
            )
            for _ in range(num_layers)
        ])
        self.output_projection = Linear(
                dim_model,
                vocab_size,
                bias = bias,
                dtype = dtype,
            )

        self.layernorm = Layernorm(dim_model, dtype=dtype)

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:
        with torch.no_grad():
            mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
            mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])
            mask_2d[:] = True

        out = self.word_emb(input)

        # for layer in self.transformers:
        out = self.transformers(out, mask_2d, self.position_bias)
        out = self.layernorm(out)

        logits = self.output_projection(out)
        bmt.inspect.record_tensor(logits, "logits")

        return logits