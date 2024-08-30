""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)


class NeuronFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["NeuronFlashAttentionBackendImpl"]:
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "NeuronFlashAttentionMetadata":
        return NeuronFlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise NotImplementedError


@dataclass
class NeuronFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for NeuronFlashAttentionBackend.
    """

    # inherit from AttentionMetadata
    # num_prefills: int
    # num_prefill_tokens: int
    # num_decode_tokens: int
    # slot_mapping: torch.Tensor

    # inherit from PagedAttentionMetadata
    # seq_lens_tensor: Optional[torch.Tensor] = None
    # max_decode_seq_len: int = 0
    # block_tables: Optional[torch.Tensor] = None # (batch_size, max_blocks_per_seq).

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    # seq_lens_tensor: Optional[torch.Tensor] = None
    # prompt_lens_tensor: Optional[torch.Tensor] = None # deprecated - to be replaced with seq_lens_tensor
    is_prompt: bool
    slot_mapping: torch.Tensor
    seq_lens: Optional[List[int]]
    context_lens: Optional[torch.Tensor]
    # prompt_lens_tensor: Optional[torch.Tensor]

    # # FIXME: It is for flash attn.
    # # Maximum sequence length among prefill batch. 0 if there are decoding
    # # requests only.
    # max_prefill_seq_len: int = 0
    # # Maximum sequence length among decode batch. 0 if there are prefill
    # # requests only.
    # max_decode_seq_len: int = 0

    # # (batch_size,). The sequence length per sequence. Sequence length means
    # # the computed tokens + new tokens None if it is a decoding.
    # seq_lens: Optional[List[int]] = None

    # # FIXME: It is for flash attn.
    # # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # # the batch, used to index into sequence. E.g., if the sequence length is
    # # [4, 6], it is [0, 4, 10].
    # seq_start_loc: Optional[torch.Tensor] = None

    # # (batch_size,) A tensor of context lengths (tokens that are computed
    # # so far).
    # context_lens_tensor: Optional[torch.Tensor] = None
    # context_lens: Optional[torch.Tensor] = None

    # # Maximum query length in the batch. None for decoding.
    # max_query_len: Optional[int] = None

    # # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # # the batch, used to index into subquery. E.g., if the subquery length
    # # is [4, 6], it is [0, 4, 10].
    # query_start_loc: Optional[torch.Tensor] = None

    # # is_prompt: Optional[bool] = None # deprecated - to be replaced with max_prefill_seq_len
    # slot_mapping: Optional[torch.Tensor] = None
    # # num_prefills : int = None # deprecated - to be replaced with seq_lens (aka batch_size)

    # def __post_init__(self):
    #     # Set during the execution of the first attention op.
    #     # It is a list because it is needed to set per prompt
    #     # when alibi slopes is used. It is because of the limitation
    #     # from xformer API.
    #     # will not appear in the __repr__ and __init__
    #     self.attn_bias: Optional[List[torch.Tensor]] = None

    @property
    def prompt_lens_tensor(self):
        # For backward compatibility only. This is going to be deprecated.
        return seq_lens_tensor
