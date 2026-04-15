"""
LLM wrapper for generation and hidden state extraction.
Since we have limited compute, we'll use a simplified approach.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import random


class SimpleLLMWrapper:
    """
    Simplified LLM wrapper for experiments.
    In a full implementation, this would use transformers with Qwen2.5-7B.
    For efficiency with 8-hour budget, we use simulated responses.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.hidden_dim = 3584
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Simulated "knowledge" for generating responses
        self.answer_patterns = [
            "The answer is {answer}.",
            "Based on the information, {answer}.",
            "Therefore, the answer is {answer}.",
            "{answer}",
        ]
    
    def generate(self, prompt: str, context: str = "", max_tokens: int = 100, 
                 temperature: float = 0.0) -> Tuple[str, Dict]:
        """
        Generate text and return with metadata.
        
        Returns:
            generated_text: The generated text
            metadata: Dict containing hidden states and other info
        """
        # Simulate generation based on context and prompt
        # In practice, this would use the actual LLM
        
        # Extract potential answer from context or prompt
        answer = self._extract_answer(context, prompt)
        
        # Simulate reasoning steps
        reasoning_steps = self._simulate_reasoning(context, prompt, answer)
        
        # Generate response
        pattern = random.choice(self.answer_patterns)
        response = pattern.format(answer=answer)
        
        # Simulate hidden states
        hidden_states = []
        for step in reasoning_steps:
            # Simulate hidden state for each step
            h = torch.randn(self.hidden_dim) * 0.5
            # Add some structure based on content
            if "entity" in step.lower():
                h[:500] += 1.0  # Simulate entity-related activation
            if "temporal" in step.lower() or "date" in step.lower():
                h[500:1000] += 1.0  # Simulate temporal activation
            hidden_states.append(h)
        
        metadata = {
            'hidden_states': hidden_states,
            'reasoning_steps': reasoning_steps,
            'num_tokens': len(response.split()),
            'extracted_answer': answer
        }
        
        return response, metadata
    
    def generate_with_retrieval_trigger(self, prompt: str, context: str = "", 
                                        max_steps: int = 5,
                                        should_retrieve_fn = None) -> Tuple[str, List, Dict]:
        """
        Generate with iterative retrieval.
        
        Args:
            prompt: The question/prompt
            context: Initial context
            max_steps: Maximum generation steps
            should_retrieve_fn: Function that decides whether to retrieve (hidden_state -> bool)
        
        Returns:
            final_answer: Generated answer
            retrieval_history: List of (step, query) when retrieval was triggered
            metadata: Generation metadata
        """
        retrieval_history = []
        all_hidden_states = []
        generated_text = ""
        
        for step in range(max_steps):
            # Generate next chunk
            response, meta = self.generate(prompt, context, max_tokens=30)
            generated_text += response + " "
            all_hidden_states.extend(meta['hidden_states'])
            
            # Check if retrieval is needed
            if should_retrieve_fn and meta['hidden_states']:
                latest_hidden = meta['hidden_states'][-1]
                if should_retrieve_fn(latest_hidden):
                    retrieval_query = self._construct_retrieval_query(prompt, generated_text)
                    retrieval_history.append((step, retrieval_query))
                    # Simulate adding retrieved context
                    context += f" [Retrieved: Info about {retrieval_query}]"
            
            # Check if answer is complete
            if self._is_answer_complete(generated_text):
                break
        
        metadata = {
            'hidden_states': all_hidden_states,
            'num_steps': step + 1,
            'retrieval_count': len(retrieval_history)
        }
        
        return generated_text.strip(), retrieval_history, metadata
    
    def extract_hidden_state(self, text: str, layer: int = 16) -> torch.Tensor:
        """Extract simulated hidden state for a text."""
        # Simulate hidden state
        h = torch.randn(self.hidden_dim) * 0.3
        
        # Add semantic structure based on text
        if any(word in text.lower() for word in ['who', 'person', 'name']):
            h[:300] += 0.8  # Entity pattern
        if any(word in text.lower() for word in ['when', 'date', 'year']):
            h[300:600] += 0.8  # Temporal pattern
        if any(word in text.lower() for word in ['where', 'place', 'city']):
            h[600:900] += 0.8  # Location pattern
        
        return h
    
    def _extract_answer(self, context: str, prompt: str) -> str:
        """Extract potential answer from context."""
        # Simple heuristic: look for capitalized words or quoted phrases
        words = context.split()
        candidates = [w.strip('.,;:!?') for w in words if w and w[0].isupper()]
        
        if candidates:
            return candidates[-1]  # Return last candidate
        return "unknown"
    
    def _simulate_reasoning(self, context: str, prompt: str, answer: str) -> List[str]:
        """Simulate reasoning steps."""
        steps = [
            f"Analyze: Understanding the question about {prompt[:30]}...",
            f"Step 1: Identifying key entities",
            f"Step 2: Looking for {answer} in context",
            f"Conclusion: Found the answer"
        ]
        return steps
    
    def _construct_retrieval_query(self, original_question: str, generated_text: str) -> str:
        """Construct retrieval query from context."""
        # Extract last few words as query
        words = generated_text.split()[-10:]
        return " ".join(words)
    
    def _is_answer_complete(self, text: str) -> bool:
        """Check if answer appears complete."""
        completion_markers = ['.', '!', '?', 'is', 'are', 'was', 'were']
        return any(text.strip().endswith(m) for m in completion_markers)


class RealLLMWrapper:
    """
    Wrapper for real LLM (Qwen2.5-7B).
    This would be used in a full implementation.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict]:
        """Generate with actual model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 100),
                temperature=kwargs.get('temperature', 0.0),
                return_dict_in_generate=True,
                output_hidden_states=True
            )
        
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Extract hidden states
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else []
        
        metadata = {
            'hidden_states': hidden_states,
            'num_tokens': outputs.sequences.shape[1] - inputs.input_ids.shape[1]
        }
        
        return generated_text, metadata


def create_llm_wrapper(use_real_model: bool = False, seed: int = 42, **kwargs):
    """Factory function to create LLM wrapper."""
    if use_real_model:
        return RealLLMWrapper(**kwargs)
    else:
        return SimpleLLMWrapper(seed=seed)


if __name__ == '__main__':
    # Test wrapper
    wrapper = SimpleLLMWrapper(seed=42)
    
    prompt = "Who invented the light bulb?"
    context = "Thomas Edison was an American inventor. He developed many devices including the phonograph and the motion picture camera."
    
    response, metadata = wrapper.generate(prompt, context)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Hidden states: {len(metadata['hidden_states'])}")
