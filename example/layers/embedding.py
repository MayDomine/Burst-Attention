import math
from typing import Optional,Tuple
import torch
import torch.nn.functional as F
import bmtrain as bmt


class Embedding(bmt.DistributedModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[torch.Tensor] = None,
                 dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = bmt.DistributedParameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cuda"), init_method=torch.nn.init.normal_)
        else:
            self.weight = bmt.DistributedParameter(_weight)
        
        self.sparse = sparse
    
    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (boolean, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, input: torch.Tensor, projection : bool = False) -> torch.Tensor:
        if not projection:
            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse).clone()
        else:
            return F.linear(input, self.weight) / math.sqrt(self.embedding_dim)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

class RotaryEmbeddingESM(bmt.DistributedModule):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self, 
        dim: int, 
        base = 10000,
        distance_scale = 1,
        dtype = torch.half, 
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale
        self.dtype = dtype

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq.to(dtype), persistent=False)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, length, right, cos, sin):
        if cos.dim() == 2:
            cos = cos[right-length:right, :]
            sin = sin[right-length:right, :]
        elif cos.dim() == 3:
            cos = cos[:, right-length:right, :]
            sin = sin[:, right-length:right, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, right-length:right, :]
            sin = sin[:, :, right-length:right, :]
            
        return (x * cos) + (self.rotate_half(x) * sin) 

    def _update_cos_sin_tables(self, x, seq_dim):
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if x.dim() == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim= -2) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)
        return (
            self.apply_rotary_pos_emb(q, q.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached).clone(),
            self.apply_rotary_pos_emb(k, k.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached).clone(),
        )