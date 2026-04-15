"""
Model loading utilities for vLLM inference.
"""
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_model(model_name: str, gpu_memory_utilization: float = 0.85):
    """Load a model with vLLM."""
    print(f"Loading model: {model_name}")
    
    # Map model names to HuggingFace paths
    model_paths = {
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    model_path = model_paths.get(model_name, model_name)
    
    # Check if we need HF token
    hf_token = os.environ.get("HF_TOKEN", None)
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Load model with vLLM
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=8192,
        )
        
        print(f"Successfully loaded {model_name}")
        return llm, tokenizer
    
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def get_sampling_params(temperature: float = 0.0, max_tokens: int = 2048):
    """Get sampling parameters for generation."""
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0 if temperature == 0 else 0.95,
    )


class LLMWrapper:
    """Wrapper for vLLM model to provide consistent interface."""
    
    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        max_tokens: int = 2048,
        stop: list = None
    ) -> str:
        """Generate text from prompt."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0 else 0.95,
            stop=stop or [],
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_batch(
        self,
        prompts: list,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> list:
        """Generate text for multiple prompts."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        
        outputs = self.model.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]
    
    def generate_with_logprobs(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple:
        """Generate text and return with logprobs."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0 else 0.95,
            logprobs=1,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # Extract logprobs
        logprobs = []
        if hasattr(outputs[0].outputs[0], 'logprobs') and outputs[0].outputs[0].logprobs:
            for logprob_dict in outputs[0].outputs[0].logprobs:
                if logprob_dict:
                    # Get the highest logprob for each position
                    max_logprob = max(logprob_dict.values())
                    logprobs.append(max_logprob)
        
        return output_text, logprobs
