import importlib
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, CacheConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.cache_engine import CacheEngine

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


# Models supported by Neuronx distributed for inference.
_NEURON_SUPPORTED_MODELS: Dict[str, Tuple[str, str]] = {
    "LlamaForCausalLM": ("neuronx_distributed_inference.models.llama.modeling_llama",
                         "NeuronLlamaForCausalLM"),
}

class NeuronCasualLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata,
    ) -> torch.Tensor:
        print()
        print()
        # print(f"## input_ids")
        # print(input_ids.flatten())
        # print(f"## cache_ids")
        # print(positions.flatten())
        # print(f"## slot_mapping")
        # print(input_metadata.slot_mapping.flatten())
        # print(f"## prompt_lens")
        # print(input_metadata.seq_lens_tensor)
        # print(f"## block_tables")
        # print(input_metadata.block_tables.flatten())
        # print(f"## input_metadata")
        # print(input_metadata)

        output = self.model(
            input_ids.reshape(1, -1).long(),
            position_ids=positions.reshape(1, -1).long(),
            slot_mapping=input_metadata.slot_mapping.long(), 
            block_table=input_metadata.block_tables.long(),
            prompt_lens=input_metadata.seq_lens_tensor.long(),
            context_lens=input_metadata.context_lens.long(),
        )
        return output.logits[0, :, :]

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        print(f"## logits shape")
        print(logits.shape)
        # print(torch.min(logits, dim=1).values.shape)
        # print("logits", logits - torch.min(logits, dim=1, keepdim=True).values)
        # logits_subtract = logits - torch.min(logits, dim=1, keepdim=True).values
        # print("logits max", torch.max(logits_subtract, dim=1))
        next_tokens = self.sampler(logits, sampling_metadata)
        print(f"## output token")
        print(next_tokens)
        return next_tokens

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = kwargs['neuron_config']
        self.config.neuron_config = neuron_config
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        self.model = neuronx_model_cls(model_name_or_path, config)
        compiled_model_path = os.path.join(model_name_or_path,
            f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
        try:
            self.model.load(compiled_model_path)
        except ValueError:
            self.model.compile(compiled_model_path)
            self.model.load(compiled_model_path)


def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")


def determine_num_available_blocks(cache_config, model_config, parallel_config) -> Tuple[int, int]:
    """Determine the number of available KV blocks.

    Swapping is not yet supported, so always return num_cpu_blocks=0.
    """
    total_gpu_memory = 16 * 1e9 # 16GiB per NeuronCore
    cache_block_size = CacheEngine.get_cache_block_size(cache_config, model_config, parallel_config)
    num_gpu_blocks = int((total_gpu_memory * cache_config.gpu_memory_utilization) // cache_block_size)
    num_gpu_blocks = max(num_gpu_blocks, 0)
    assert num_gpu_blocks > 0, f"insufficient K/V cache space."

    # Swap not yet supported with Neuron backend.
    num_cpu_blocks = 0

    return num_gpu_blocks, num_cpu_blocks


def _get_default_neuron_config(model_config: ModelConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig,
                               cache_config: CacheConfig):
    max_blocks_per_seq = scheduler_config.max_model_len // cache_config.block_size

    num_gpu_blocks, num_cpu_blocks = determine_num_available_blocks(
        cache_config, model_config, parallel_config
    )

    if cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks = cache_config.num_gpu_blocks_override
    
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks

    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        batch_size=1,
        max_context_length=scheduler_config.max_num_batched_tokens,
        seq_len=scheduler_config.max_model_len,

        is_paged_attention=True,
        pa_num_blocks=num_gpu_blocks,
        pa_block_size=cache_config.block_size,
        is_chunked_prefill=True,
        cf_max_num_seqs=scheduler_config.max_num_seqs,
        # max_num_seqs * max_blocks_per_seq = 8*8=64
        cf_num_active_blocks=scheduler_config.max_num_seqs * max_blocks_per_seq,

        enable_bucketing=False,
        is_continuous_batching=False,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        padding_side="right"
    )
    return neuron_config

def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    from neuronx_distributed_inference.models.config import NeuronConfig
    overridden_neuron_config = overridden_neuron_config or {}
    default_neuron_config.update(overridden_neuron_config)
    return NeuronConfig(**default_neuron_config)

def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig,
                     cache_config: CacheConfig) -> nn.Module:
    model = NeuronCasualLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_config(
        model_config, parallel_config, scheduler_config, cache_config)
    neuron_config = _get_neuron_config_after_override(default_neuron_config_args,
        None)
    model.load_weights(model_config.model,
                       neuron_config=neuron_config,)
    return model.eval()