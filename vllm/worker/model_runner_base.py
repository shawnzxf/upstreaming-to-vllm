import dataclasses
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Set, Type,
                    TypeVar)

import torch

from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.multimodal import MultiModalInputs
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.model_executor import SamplingMetadata

T = TypeVar('T', bound="BroadcastableModelInput")


def _add_attn_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Any],
        attn_metadata: Optional["AttentionMetadata"]) -> None:
    """
    Helper method to update tensor_dict with broadcastable
    AttentionMetadata fields.
    """
    if attn_metadata is not None:
        tensor_dict.update(attn_metadata.asdict_zerocopy())


def _init_attn_metadata_from_tensor_dict(
    attn_backend: "AttentionBackend",
    tensor_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper method to initialize AttentionMetadata based on an
    AttentionBackend and broadcastable AttentionMetadata fields.
    """
    # Extract the fields used to create AttentionMetadata.
    valid_attn_kwargs = {}
    for field in dataclasses.fields(attn_backend.get_metadata_cls()):
        val = tensor_dict.pop(field.name, None)
        if val is not None:
            valid_attn_kwargs[field.name] = val

    attn_metadata = attn_backend.make_metadata(**valid_attn_kwargs)
    tensor_dict["attn_metadata"] = attn_metadata
    return tensor_dict


def _init_sampling_metadata_from_tensor_dict(  # type: ignore
        tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper method to initialize SamplingMetadata based on broadcastable
    SamplingMetadata fields.
    """
    from vllm.model_executor import SamplingMetadata

    selected_token_indices = tensor_dict.pop("selected_token_indices", None)
    # An empty SamplingMetadata to signal that the worker should skip
    # sampling.
    if selected_token_indices is not None:
        tensor_dict["sampling_metadata"] = SamplingMetadata(
            seq_groups=None,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=None,
            num_prompts=0,
        )
    return tensor_dict


def _add_sampling_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Any],
        sampling_metadata: Optional["SamplingMetadata"]) -> None:
    """
    Helper method to update tensor_dict with broadcastable
    SamplingMetadata fields.
    """
    if sampling_metadata is not None:
        tensor_dict["selected_token_indices"] = (
            sampling_metadata.selected_token_indices)


def _init_frozen_model_input_from_tensor_dict(
        frozen_model_input_cls: Type["ModelRunnerInputBase"],
        tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper method to initialize a frozen ModelInput based on broadcastable
    """
    valid_tensor_kwargs = {}
    for field in dataclasses.fields(frozen_model_input_cls):
        val = tensor_dict.pop(field.name, None)
        if val is not None:
            valid_tensor_kwargs[field.name] = val

    frozen_model_input = frozen_model_input_cls(**valid_tensor_kwargs)
    tensor_dict["frozen_model_input"] = frozen_model_input
    return tensor_dict


class BroadcastableModelInput(ABC):

    @abstractmethod
    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_broadcasted_tensor_dict(
        cls: Type[T],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> T:
        """
        Pop fields from the given tensor_dict and populate a new instance of
        BroadcastableModelInput.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ModelRunnerInputBase(BroadcastableModelInput):
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelRunnerInputBase objects.

    Model runners that support multi-GPU execution should define a
    ModelRunnerInputBase subclass, add their required fields, and specify how to
    serialize/deserialize a ModelInput for broadcast between workers.
    """
    pass


class ModelRunnerInputBuilderBase(ABC, Generic[T]):
    """A builder to create ModelRunnerInputBase objects.
  """

    @abstractmethod
    def add_seq_group(self, seq_group_metadata):
        """TBA"""
        raise NotImplementedError

    @abstractmethod
    def build(self, *args, **kwargs) -> T:
        """Build metadata with on-device tensors."""
        raise NotImplementedError


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding
    ModelRunnerInputBase subclass.
    """

    # Map of request_id -> generator used for seeded random sampling
    generators: Dict[str, torch.Generator] = {}

    @abstractmethod
    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> T:
        """
        Make an instance of a ModelRunnerInputBase from the broadcasted tensor
        dict.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> T:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @current_platform.inference_mode()
    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
        intermediate_tensors: Optional[IntermediateTensors],
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError

    def get_generators(self, finished_request_ids: Optional[List[str]] = None):
        """
        Return dict of per-request generators used for random sampling.
        """

        # Clean up generators from completed requests
        if finished_request_ids:
            for request_id in finished_request_ids:
                self.generators.pop(request_id, None)

        return self.generators


# Note: ideally we would be using a dataclass(kw_only=True)
# here, so that this can be subclassed easily,
# but kw_only is not supported in python<3.10.
class InterDataForSeqGroup:
    """Intermediate data for the current sequence group."""

    def simple_reinit(self):
        self.input_tokens[0].clear()  # type: ignore
        self.input_positions[0].clear()  # type: ignore
        self.seq_lens[0] = 0  # type: ignore
        self.orig_seq_lens[0] = 0  # type: ignore
        self.query_lens[0] = 0  # type: ignore
        self.context_lens[0] = 0  # type: ignore
        self.curr_sliding_window_blocks[0] = 0  # type: ignore
        self.lora_index_mapping.clear()  # type: ignore
        self.lora_prompt_mapping.clear()  # type: ignore
        self.lora_requests.clear()  # type: ignore
        self.prompt_adapter_index_mapping.clear()  # type: ignore
        self.prompt_adapter_prompt_mapping.clear()  # type: ignore

    def __init__(
        self,
        *,
        # From sequence group metadata.
        request_id: str,
        seq_ids: List[int],
        is_prompt: bool,
        block_tables: Optional[Dict[int, List[int]]],
        computed_block_nums: List[int],
        n_seqs: int = 0,

        # Input tokens and positions.
        input_tokens: Optional[List[List[int]]] = None,
        input_positions: Optional[List[List[int]]] = None,

        # The sequence length (may be capped to the sliding window).
        seq_lens: Optional[List[int]] = None,
        # The original sequence length (before applying sliding window).
        # This is used to compute slot mapping.
        orig_seq_lens: Optional[List[int]] = None,
        # The query length.
        query_lens: Optional[List[int]] = None,
        # The number of tokens that are already computed.
        context_lens: Optional[List[int]] = None,
        # The current sliding window block.
        curr_sliding_window_blocks: Optional[List[int]] = None,

        # LoRA inputs.
        lora_index_mapping: Optional[List[List[int]]] = None,
        lora_prompt_mapping: Optional[List[List[int]]] = None,
        lora_requests: Optional[Set[LoRARequest]] = None,

        # Prompt adapter inputs.
        prompt_adapter_index_mapping: Optional[List[int]] = None,
        prompt_adapter_prompt_mapping: Optional[List[int]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,

        # Multi-modal inputs.
        multi_modal_inputs: Optional[MultiModalInputs] = None,

        # Whether the prefix cache is hit (prefill only).
        prefix_cache_hit: bool = False,
        reinit: bool = False,
        reinit_use_defaults: bool = False,
    ):
        if reinit:
            assert len(self.seq_ids) == len(seq_ids)  # type: ignore
            for i, seq_id in enumerate(seq_ids):
                self.seq_ids[i] = seq_id  # type: ignore
        else:
            self.seq_ids = seq_ids

        self.request_id = request_id
        self.is_prompt = is_prompt
        self.block_tables = block_tables
        self.computed_block_nums = computed_block_nums
        self.n_seqs = n_seqs

        if reinit:
            if len(self.seq_ids) == 1 and reinit_use_defaults:
                self.simple_reinit()
            else:
                if input_tokens:
                    self.input_tokens = input_tokens
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.input_tokens[seq_id].clear()

                if input_positions:
                    self.input_positions = input_positions
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.input_positions[seq_id].clear()

                if seq_lens:
                    self.seq_lens = seq_lens
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.seq_lens[seq_id] = 0

                if orig_seq_lens:
                    self.orig_seq_lens = orig_seq_lens
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.orig_seq_lens[seq_id] = 0

                if query_lens:
                    self.query_lens = query_lens
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.query_lens[seq_id] = 0

                if context_lens:
                    self.context_lens = context_lens
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.context_lens[seq_id] = 0

                if curr_sliding_window_blocks:
                    self.curr_sliding_window_blocks = \
                        curr_sliding_window_blocks
                else:
                    for seq_id in range(len(self.seq_ids)):
                        self.curr_sliding_window_blocks[seq_id] = 0

                if lora_index_mapping:
                    self.lora_index_mapping = lora_index_mapping
                else:
                    self.lora_index_mapping.clear()

                if lora_prompt_mapping:
                    self.lora_prompt_mapping = lora_prompt_mapping
                else:
                    self.lora_prompt_mapping.clear()

                if lora_requests:
                    self.lora_requests = lora_requests
                else:
                    self.lora_requests.clear()

                if prompt_adapter_index_mapping:
                    self.prompt_adapter_index_mapping = \
                        prompt_adapter_index_mapping
                else:
                    self.prompt_adapter_index_mapping.clear()

                if prompt_adapter_prompt_mapping:
                    self.prompt_adapter_prompt_mapping = \
                        prompt_adapter_prompt_mapping
                else:
                    self.prompt_adapter_prompt_mapping.clear()

        else:
            self.input_tokens = input_tokens or []
            self.input_positions = input_positions or []
            self.seq_lens = seq_lens or []
            self.orig_seq_lens = orig_seq_lens or []
            self.query_lens = query_lens or []
            self.context_lens = context_lens or []
            self.curr_sliding_window_blocks = \
                curr_sliding_window_blocks or []

            self.lora_index_mapping = lora_index_mapping or []
            self.lora_prompt_mapping = lora_prompt_mapping or []
            self.lora_requests = lora_requests or set()

            self.prompt_adapter_index_mapping = (
                prompt_adapter_index_mapping or [])
            self.prompt_adapter_prompt_mapping = (
                prompt_adapter_prompt_mapping or [])

        self.prompt_adapter_request = prompt_adapter_request
        self.multi_modal_inputs = multi_modal_inputs
        self.prefix_cache_hit = prefix_cache_hit

        self.n_seqs = len(self.seq_ids)

        if not reinit:
            self.__post_init__()

    def __post_init__(self):
        self.n_seqs = len(self.seq_ids)

        self.input_tokens = [[] for _ in range(self.n_seqs)]
        self.input_positions = [[] for _ in range(self.n_seqs)]
        self.seq_lens = [0] * self.n_seqs
        self.orig_seq_lens = [0] * self.n_seqs
        self.query_lens = [0] * self.n_seqs
        self.context_lens = [0] * self.n_seqs
        self.curr_sliding_window_blocks = [0] * self.n_seqs

        self.lora_index_mapping = []
        self.lora_prompt_mapping = []
