import os

from vllm import LLM, SamplingParams

# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows",
    "The future of AI is",
    "Hello, my name can",
    "The president of the United States can",
    "The capital of France can",
    "The future of AI can",
    "The sun rises in the",
    "The quick brown fox",
    "The biggest city in India is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=10, top_k=1)

# Create an LLM.
llm = LLM(
    # model="nickypro/tinyllama-15M",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # model="openlm-research/open_llama_3b",
    # model="/shared_3/chndkv/llama-models/Meta-Llama-3.1-8B-Instruct/",
    tensor_parallel_size=32,
    max_num_seqs=8,

    max_model_len=256,
    max_num_batched_tokens=64,
    enable_chunked_prefill=True,

    # max_model_len=256,
    # max_num_batched_tokens=256,
    # enable_chunked_prefill=True,

    block_size=32,
    # gpu_memory_utilization=0.05,
    num_gpu_blocks_override=128,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

tokenizer = llm.get_tokenizer()
for i, prompt in enumerate(prompts):
    input_ids = tokenizer.encode(prompt, return_tensors="pt") 
    num_input_tokens = len(input_ids[0])
    print(f"prompt {i}, num_input_tokens: {num_input_tokens}")


# # Sample prompts.
# prompts = [
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(top_k=1)

# # Create an LLM.
# llm = LLM(
#     # TODO: Model name unsupported with neuronx-distributed framework.
#     model="/home/ubuntu/Nxd/models/Llama-2-7b",
#     max_num_seqs=4,
#     # The max_model_len and block_size arguments are required to be same as
#     # max sequence length when targeting neuron device.
#     # Currently, this is a known limitation in continuous batching support
#     # in neuronx-distributed-inference.
#     # TODO: Support paged-attention
#     max_model_len=128,
#     block_size=128,
#     # The device can be automatically detected when AWS Neuron SDK is installed.
#     # The device argument can be either unspecified for automated detection,
#     # or explicitly assigned.
#     device="neuron",
#     tensor_parallel_size=32)
# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")