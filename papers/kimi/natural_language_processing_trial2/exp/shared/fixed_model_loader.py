"""
Fixed model loading utilities with GPU memory optimizations.
Uses lower gpu_memory_utilization and proper error handling.
"""
import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# Model path mapping
MODEL_PATHS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def get_model_path(model_name: str) -> str:
    """Get HuggingFace path for model name."""
    return MODEL_PATHS.get(model_name, model_name)


class SimpleTransformersBackend:
    """Simple backend using transformers with device_map='auto'."""
    
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_name = model_name
        self.model_path = get_model_path(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_name} from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        print(f"Model loaded. Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, stop: list = None) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in text:
                    text = text[:text.index(stop_seq)]
        
        return text.strip()
    
    def generate_batch(self, prompts: list, temperature: float = 0.0, max_tokens: int = 2048) -> list:
        """Generate for multiple prompts."""
        return [self.generate(p, temperature, max_tokens) for p in prompts]
    
    def generate_with_logprobs(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> tuple:
        """Generate with logprobs for confidence estimation."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs.sequences[0][input_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Extract logprobs
        logprobs = []
        for score in outputs.scores:
            probs = torch.softmax(score, dim=-1)
            top_prob = probs.max().item()
            logprobs.append(torch.log(torch.tensor(top_prob)).item())
        
        return text.strip(), logprobs


class VLLMBackend:
    """vLLM backend with reduced memory utilization."""
    
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.70):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        self.model_name = model_name
        self.model_path = get_model_path(model_name)
        
        print(f"Loading {model_name} with vLLM (gpu_util={gpu_memory_utilization})...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load model with reduced memory utilization
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=4096,
        )
        
        print(f"Model loaded successfully")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, stop: list = None) -> str:
        """Generate text from prompt."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0 else 0.95,
            stop=stop or [],
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def generate_batch(self, prompts: list, temperature: float = 0.0, max_tokens: int = 2048) -> list:
        """Generate for multiple prompts."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        
        outputs = self.model.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
    
    def generate_with_logprobs(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> tuple:
        """Generate with logprobs."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0 else 0.95,
            logprobs=1,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text.strip()
        
        # Extract logprobs
        logprobs = []
        if hasattr(outputs[0].outputs[0], 'logprobs') and outputs[0].outputs[0].logprobs:
            for logprob_dict in outputs[0].outputs[0].logprobs:
                if logprob_dict:
                    max_logprob = max(logprob_dict.values())
                    logprobs.append(max_logprob)
        
        return output_text, logprobs


def load_model(model_name: str, backend: str = "auto", gpu_memory_utilization: float = 0.70):
    """
    Load a model with the specified backend.
    
    Args:
        model_name: Short name or full path
        backend: 'vllm', 'transformers', or 'auto'
        gpu_memory_utilization: GPU memory fraction for vLLM
    
    Returns:
        Model backend instance
    """
    # Try vLLM first if backend is auto or vllm
    if backend in ("auto", "vllm"):
        try:
            return VLLMBackend(model_name, gpu_memory_utilization)
        except Exception as e:
            print(f"vLLM failed: {e}")
            if backend == "vllm":
                raise
            print("Falling back to transformers backend...")
    
    # Use transformers backend
    return SimpleTransformersBackend(model_name)


class LLMWrapper:
    """Unified wrapper for both backends."""
    
    def __init__(self, backend, model_name: str):
        self.backend = backend
        self.model_name = model_name
        self.tokenizer = backend.tokenizer
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, stop: list = None) -> str:
        return self.backend.generate(prompt, temperature, max_tokens, stop)
    
    def generate_batch(self, prompts: list, temperature: float = 0.0, max_tokens: int = 2048) -> list:
        return self.backend.generate_batch(prompts, temperature, max_tokens)
    
    def generate_with_logprobs(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> tuple:
        return self.backend.generate_with_logprobs(prompt, temperature, max_tokens)


if __name__ == "__main__":
    # Test loading
    print("Testing model loader...")
    model = load_model("llama-3.1-8b", backend="transformers")
    wrapper = LLMWrapper(model, "llama-3.1-8b")
    
    # Test generation
    response = wrapper.generate("What is 2+2? Answer briefly.")
    print(f"Test response: {response}")
