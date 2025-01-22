import os
from vllm import LLM, SamplingParams


# 1b
MODEL_PATH = "/home/ubuntu/models/llama-3.2-1b-Instruct/"
COMPILED_MODEL_PATH = "/home/ubuntu/traced_models/llama-3.2-1b-instruct-cp-vllm/"

# Use neuronx-distributed-inference framework over transformers-neuronx
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_COMPILED_ARTIFACTS'] = COMPILED_MODEL_PATH

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
    model=MODEL_PATH,
    tensor_parallel_size=32,
    max_num_seqs=8,

    max_model_len=2048,
    max_num_batched_tokens=256, # chunk size
    enable_chunked_prefill=True,

    block_size=32,
    # gpu_memory_utilization=0.05,
    num_gpu_blocks_override=512,
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