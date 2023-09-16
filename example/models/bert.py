import torch
import bmtrain as bmt
from layers import TransformerEncoder, Layernorm, Embedding, TransformerEncoder, RotaryEmbeddingESM

class Bert(bmt.DistributedModule):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None,sequence_parallel : bool = False,flash: bool = False,
            sequence_parallel_impl="burst",gated=False,pos_bias_type="none"
        ) -> None:
        super().__init__()

        self.max_distance = max_distance

        self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        zero_level=3
        checkpointing=False,
        if pos_bias_type == "rotary":
            self.position_bias = RotaryEmbeddingESM(
            dim=dim_head,
            dtype=dtype,
            )
        self.pos_bias_type = pos_bias_type
        self.transformers = bmt.TransformerBlockList([
            bmt.Block(
                TransformerEncoder(
                    dim_model, dim_head, num_heads, dim_ff, bias, dtype,sequence_parallel,flash,sequence_parallel_impl,gated=gated,pos_bias_type=pos_bias_type
               ),zero_level=zero_level, use_checkpoint=checkpointing
            )
            for _ in range(num_layers)
        ])

        self.layernorm = Layernorm(dim_model, dtype=dtype)

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:

        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])
        mask_2d[:] = True
        if self.pos_bias_type == "rotary":
            out = self.word_emb(input)
            out = self.transformers(out, mask_2d, self.position_bias)
        else:
            out = self.pos_emb(pos) + self.word_emb(input)
            out = self.transformers(out, mask_2d, None)
        # for layer in self.transformers:

        out = self.layernorm(out)

        logits = self.word_emb(out, projection=True)
        bmt.inspect.record_tensor(logits, "logits")

        return logits