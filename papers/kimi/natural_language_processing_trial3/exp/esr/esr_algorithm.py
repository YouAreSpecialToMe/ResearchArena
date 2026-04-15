"""ESR (Entropy-guided Stepwise Revision) Algorithm Implementation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import re


class ESRGenerator:
    """Generator with entropy-guided stepwise revision."""
    
    def __init__(
        self,
        model,
        tokenizer,
        tau_h: float = 2.5,
        tau_v: float = 1.5,
        r_max: int = 3,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.tau_v = tau_v
        self.r_max = r_max
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.revision_count = 0
        
    def compute_uncertainty(self, logits: torch.Tensor) -> Tuple[float, float]:
        """Compute entropy and varentropy."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        entropy = -(probs * log_probs).sum()
        varentropy = (probs * (log_probs + entropy) ** 2).sum()
        
        return entropy.item(), varentropy.item()
    
    def should_revise(self, entropy: float, varentropy: float) -> bool:
        """Check if revision should be triggered."""
        return entropy > self.tau_h and varentropy < self.tau_v
    
    def find_step_boundaries(self, text: str) -> List[int]:
        """Find reasoning step boundaries in text."""
        boundaries = []
        lines = text.split('\n')
        pos = 0
        
        step_patterns = [
            r'^Step\s+\d+',
            r'^\d+[.):\-]\s',
            r'^[-*]\s',
            r'^(First|Second|Third|Next|Then|Finally|Therefore|So|Thus)[,:]',
        ]
        
        for i, line in enumerate(lines):
            if i > 0:
                stripped = line.strip()
                for pattern in step_patterns:
                    if re.match(pattern, stripped, re.IGNORECASE):
                        boundaries.append(pos)
                        break
            pos += len(line) + 1
        
        return boundaries
    
    def get_current_step_text(self, text: str) -> str:
        """Extract the current reasoning step."""
        boundaries = self.find_step_boundaries(text)
        if not boundaries:
            # Return last sentence or line
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return sentences[-1] if sentences else text
        
        last_boundary = boundaries[-1]
        return text[last_boundary:].strip()
    
    def generate_revision(
        self,
        prefix: str,
        uncertain_step: str,
        max_tokens: int = 256
    ) -> str:
        """Generate a revision of the uncertain step."""
        revision_prompt = (
            f"{prefix}\n\n"
            f"Wait, let me reconsider this step more carefully. "
            f"The previous approach may have an issue. "
            f"Let me think through this again step by step.\n"
        )
        
        inputs = self.tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        revision = generated[len(revision_prompt):].strip()
        
        # Take only the first paragraph/step of revision
        revision_lines = revision.split('\n')
        revision = '\n'.join(revision_lines[:3])  # Take first few lines
        
        return revision
    
    def generate(
        self,
        prompt: str,
        track_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """Generate with ESR."""
        self.revision_count = 0
        revision_history = []
        uncertainty_triggers = []
        
        # Initial generation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        total_tokens = 0
        
        step_start_pos = generated_ids.shape[1]
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids,
                    return_dict=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                
                # Compute uncertainty
                entropy, varentropy = self.compute_uncertainty(next_token_logits[0])
                
                # Check for revision trigger
                if (track_uncertainty and 
                    self.should_revise(entropy, varentropy) and 
                    self.revision_count < self.r_max and
                    step > 10):  # Don't trigger too early
                    
                    # Get current generated text
                    current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reasoning_so_far = current_text[len(prompt):]
                    
                    # Check if we're at a good revision point (end of a step)
                    current_step = self.get_current_step_text(reasoning_so_far)
                    
                    if len(current_step) > 20:  # Only revise if we have substantial content
                        uncertainty_triggers.append({
                            "position": step,
                            "entropy": entropy,
                            "varentropy": varentropy,
                            "step_text": current_step[:100]
                        })
                        
                        # Generate revision
                        prefix = prompt + reasoning_so_far[:-(len(current_step))]
                        revision = self.generate_revision(prefix, current_step)
                        
                        # Replace the uncertain step
                        revision_text = reasoning_so_far[:-(len(current_step))] + revision
                        
                        # Re-tokenize from the revision point
                        new_prompt = prompt + revision_text
                        new_inputs = self.tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=2048)
                        new_inputs = {k: v.to(self.device) for k, v in new_inputs.items()}
                        
                        generated_ids = new_inputs["input_ids"]
                        self.revision_count += 1
                        revision_history.append({
                            "step": step,
                            "entropy": entropy,
                            "varentropy": varentropy,
                            "original": current_step[:100],
                            "revision": revision[:100]
                        })
                        
                        continue
                
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                total_tokens += 1
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Safety check for length
                if generated_ids.shape[1] > 4096:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        return {
            "output": output_text,
            "total_tokens": total_tokens,
            "revision_count": self.revision_count,
            "revision_history": revision_history,
            "uncertainty_triggers": uncertainty_triggers,
            "prompt": prompt
        }


class ESRSimpleGenerator(ESRGenerator):
    """Simplified ESR for efficiency - revision at end of step only."""
    
    def generate(self, prompt: str, track_uncertainty: bool = True) -> Dict[str, Any]:
        """Generate with simplified ESR."""
        self.revision_count = 0
        revision_history = []
        
        # First pass: generate full response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        uncertainty_readings = []
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids,
                    return_dict=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                entropy, varentropy = self.compute_uncertainty(next_token_logits[0])
                
                # Track uncertainty
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                token_str = self.tokenizer.decode(next_token.item())
                
                uncertainty_readings.append({
                    "entropy": entropy,
                    "varentropy": varentropy,
                    "token": token_str,
                    "position": step
                })
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if generated_ids.shape[1] > 4096:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        # Check if we should revise based on average uncertainty
        avg_entropy = sum(u["entropy"] for u in uncertainty_readings) / len(uncertainty_readings)
        avg_varentropy = sum(u["varentropy"] for u in uncertainty_readings) / len(uncertainty_readings)
        
        # Find high-uncertainty segments
        high_uncertainty_positions = [
            i for i, u in enumerate(uncertainty_readings)
            if u["entropy"] > self.tau_h and u["varentropy"] < self.tau_v
        ]
        
        # Perform revision if triggered
        if high_uncertainty_positions and self.revision_count < self.r_max:
            # Trigger revision
            revision_prompt = (
                f"{prompt}{output_text}\n\n"
                f"Wait, let me reconsider some steps more carefully. "
                f"I think there might be an issue with my reasoning. "
                f"Let me work through this again from the beginning.\n"
            )
            
            rev_inputs = self.tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048)
            rev_inputs = {k: v.to(self.device) for k, v in rev_inputs.items()}
            
            with torch.no_grad():
                rev_outputs = self.model.generate(
                    **rev_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            rev_text = self.tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
            revision_output = rev_text[len(revision_prompt):]
            
            self.revision_count = 1
            revision_history.append({
                "trigger": "high_uncertainty",
                "avg_entropy": avg_entropy,
                "avg_varentropy": avg_varentropy,
                "high_uncertainty_count": len(high_uncertainty_positions)
            })
            
            return {
                "output": revision_output,
                "total_tokens": generated_ids.shape[1] - inputs["input_ids"].shape[1] + rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1],
                "revision_count": 1,
                "revision_history": revision_history,
                "uncertainty_triggers": [{"avg_entropy": avg_entropy, "avg_varentropy": avg_varentropy}],
                "prompt": prompt,
                "initial_output": output_text
            }
        
        return {
            "output": output_text,
            "total_tokens": generated_ids.shape[1] - inputs["input_ids"].shape[1],
            "revision_count": 0,
            "revision_history": [],
            "uncertainty_triggers": [],
            "prompt": prompt
        }
