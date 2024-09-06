""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadataBuilder,
                                              AttentionMetadata)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.attention.backends.utils import (compute_slot_mapping,
                                           compute_slot_mapping_start_idx)
from vllm.utils import get_kv_cache_torch_dtype, make_tensor_with_pad


class NeuronAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["NeuronAttentionBackendImpl"]:
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "NeuronAttentionMetadata":
        return NeuronAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_builder_cls() -> Type["NeuronAttentionMetadataBuilder"]:
        return NeuronAttentionMetadataBuilder

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
class NeuronAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for NeuronAttentionBackend.
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
    is_prompt: Optional[bool]
    slot_mapping: Optional[torch.Tensor]
    seq_lens: Optional[List[int]]
    context_lens: Optional[torch.Tensor]
    # prompt_lens_tensor: Optional[torch.Tensor]

    # # FIXME: It is for flash attn.
    # # Maximum sequence length among prefill batch. 0 if there are decoding
    # # requests only.
    max_prefill_seq_len: int = 0
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


class NeuronAttentionMetadataBuilder(AttentionMetadataBuilder[NeuronAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForNeuronBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = (
            input_builder.scheduler_config.use_v2_block_manager)

        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        self.paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        self.paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        self.paged_kv_last_page_len: List[int] = []

        self.is_profile_run: bool = False

    def _add_seq_group(
            self, inter_data: "InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # is_profile_run = is_block_tables_empty(block_tables)
            is_profile_run = False

            # Compute slot mapping.
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window,
                self.use_v2_block_manager)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

            # It is not necessary to add paged_kv_indices, paged_kv_indptr,
            # and paged_kv_last_page_len for profile run because we will
            # create dummy inputs.
            # if is_profile_run:
            #     self.is_profile_run = is_profile_run
            #     return

            block_table = block_tables[seq_id]
            self._update_paged_kv_tensors(block_table, seq_len)

    def _update_paged_kv_tensors(self, block_table: List[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        block_table_bound = seq_len // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else seq_len // self.block_size
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_len.append(last_page_len)

    def build(self, seq_lens: List[int], query_lens: List[int],
              batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        # use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        # if use_captured_graph:
        #     self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
        #     self.block_tables.extend([] * cuda_graph_pad_size)
        #     num_decode_tokens = batch_size

        #     # The shape of graph_block_tables is
        #     # [max batch size, max context len // block size].
        #     input_block_tables = self.runner.graph_block_tables[:batch_size]
        #     for i, block_table in enumerate(self.block_tables):
        #         if block_table:
        #             input_block_tables[i, :len(block_table)] = block_table
        #     block_tables = torch.from_numpy(input_block_tables).to(
        #         device, non_blocking=True)

        #     last_paged_kv_indptr = self.paged_kv_indptr[-1]
        #     self.paged_kv_indptr.extend([last_paged_kv_indptr] *
        #                                 cuda_graph_pad_size)
        #     self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
        # else:
        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int,
            device=device,
        )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
        query_lens_tensor = torch.tensor(query_lens, dtype=torch.long, device=device)
        slot_mapping_tensor = torch.tensor(self.slot_mapping, dtype=torch.long, device=device)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        if len(self.paged_kv_indptr) > 0:
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device="cpu",
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device="cpu",
                                                  dtype=torch.int)
            paged_kv_last_page_len_tensor = torch.tensor(
                self.paged_kv_last_page_len, device="cpu", dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_len_tensor = None

        kv_cache_dtype = get_kv_cache_torch_dtype(
            self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device="cpu")
        query_lens_tensor = torch.tensor(query_lens, dtype=torch.int, device="cpu")
        context_lens_tensor = seq_lens_tensor - query_lens_tensor
        return NeuronAttentionMetadata(

            # from AttentionMetadata
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping_tensor,

            # from PagedAttentionMetadata
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=0,
            block_tables=block_tables,

            max_prefill_seq_len=max_prefill_seq_len,

            is_prompt=True,
            seq_lens=seq_lens_tensor,
            context_lens=context_lens_tensor,


            # paged_kv_indptr=paged_kv_indptr_tensor,
            # paged_kv_indices=paged_kv_indices_tensor,
            # paged_kv_last_page_len=paged_kv_last_page_len_tensor,
            # num_qo_heads=self.runner.model_config.get_num_attention_heads(
            #     self.runner.parallel_config),
            # num_kv_heads=self.runner.model_config.get_num_kv_heads(
            #     self.runner.parallel_config),
            # head_dim=self.runner.model_config.get_head_size(),
            # page_size=self.block_size,
            # seq_start_loc=seq_start_loc,
            # query_start_loc=query_start_loc,
            # device=device,
            # data_type=kv_cache_dtype,
            # is_profile_run=self.is_profile_run
        )

