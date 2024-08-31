import os, torch
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache")
os.environ["NEURON_CC_FLAGS"]= " -O1 "
os.environ["NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK"] = "0"
os.environ["NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK"] = "0"

from vllm import LLM, SamplingParams

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
]
# Create a sampling params object.
sampling_params = SamplingParams()

# Create an LLM.
llm = LLM(
    model="nickypro/tinyllama-15M",
    # model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # model="openlm-research/open_llama_3b",
    tensor_parallel_size=2,
    max_num_seqs=4,

    max_model_len=256,
    max_num_batched_tokens=64,
    enable_chunked_prefill=True,

    block_size=32,
    gpu_memory_utilization=0.0005)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
