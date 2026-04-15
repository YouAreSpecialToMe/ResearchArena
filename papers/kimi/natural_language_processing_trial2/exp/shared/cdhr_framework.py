"""
CDHR (Confidence-Dynamic Heterogeneous Reasoning) Framework
Core implementation of confidence dynamics monitoring and strategy selection.
"""
import numpy as np
import re
import json
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    LINEAR = "linear"
    ANALOGICAL = "analogical"
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"


@dataclass
class ReasoningStep:
    """Single reasoning step with confidence tracking."""
    text: str
    token_confidence: float = 0.0
    consistency_confidence: float = 0.0
    composite_confidence: float = 0.0
    strategy: Strategy = Strategy.LINEAR
    timestamp: float = field(default_factory=lambda: 0.0)


@dataclass
class ConfidenceDynamics:
    """Confidence dynamics metrics."""
    velocity: float = 0.0
    variance: float = 0.0
    trajectory_type: str = "unknown"  # progressing, stagnant, oscillating, declining


class CDHRSystem:
    """Confidence-Dynamic Heterogeneous Reasoning System."""
    
    def __init__(
        self,
        llm_engine,
        theta_v: float = 0.05,
        theta_sigma: float = 0.1,
        beta: float = 0.5,
        window_size: int = 3,
        max_switches: int = 5,
        retrieval_index: Optional[Dict] = None,
    ):
        self.llm = llm_engine
        self.theta_v = theta_v
        self.theta_sigma = theta_sigma
        self.beta = beta
        self.window_size = window_size
        self.max_switches = max_switches
        self.retrieval_index = retrieval_index
        
        # Strategy prompts
        self.prompts = {
            Strategy.LINEAR: "Let's solve this step by step.",
            Strategy.ANALOGICAL: "This problem resembles a similar one. Let's use analogical reasoning.",
            Strategy.DECOMPOSITION: "This is complex. Let's break it into smaller sub-problems.",
            Strategy.VERIFICATION: "Let me verify my reasoning so far and check for errors.",
        }
        
    def compute_token_confidence(self, logprobs: List[float]) -> float:
        """Compute token-level confidence from log probabilities."""
        if not logprobs:
            return 0.0
        avg_logprob = np.mean(logprobs)
        return np.exp(avg_logprob)
    
    def compute_consistency_confidence(
        self, 
        context: str, 
        current_step: str,
        k: int = 5
    ) -> float:
        """Compute self-consistency confidence by sampling continuations."""
        if self.beta == 1.0:  # Skip if only using token confidence
            return 0.0
        
        # Sample k continuations
        samples = []
        for _ in range(k):
            continuation = self.llm.generate(
                context + current_step,
                temperature=0.7,
                max_tokens=50
            )
            samples.append(continuation)
        
        # Extract answers and compute agreement
        answers = [self._extract_quick_answer(s) for s in samples]
        answer_counts = {}
        for ans in answers:
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        
        if not answer_counts:
            return 0.0
        
        max_count = max(answer_counts.values())
        return max_count / k
    
    def compute_composite_confidence(
        self,
        token_conf: float,
        consistency_conf: float
    ) -> float:
        """Compute composite confidence score."""
        return self.beta * token_conf + (1 - self.beta) * consistency_conf
    
    def compute_dynamics(
        self,
        confidence_history: List[float]
    ) -> ConfidenceDynamics:
        """Compute confidence dynamics (velocity and variance)."""
        if len(confidence_history) < self.window_size:
            return ConfidenceDynamics()
        
        # Get recent window
        recent = confidence_history[-self.window_size:]
        
        # Compute velocity (rate of change)
        if len(recent) >= 2:
            velocity = (recent[-1] - recent[0]) / (len(recent) - 1)
        else:
            velocity = 0.0
        
        # Compute variance (stability)
        variance = np.var(recent)
        
        # Classify trajectory
        trajectory_type = self._classify_trajectory(velocity, variance)
        
        return ConfidenceDynamics(
            velocity=velocity,
            variance=variance,
            trajectory_type=trajectory_type
        )
    
    def _classify_trajectory(self, velocity: float, variance: float) -> str:
        """Classify confidence trajectory type."""
        if variance > self.theta_sigma:
            return "oscillating"
        elif velocity > self.theta_v:
            return "progressing"
        elif velocity < -self.theta_v:
            return "declining"
        else:
            return "stagnant"
    
    def select_strategy(self, dynamics: ConfidenceDynamics) -> Strategy:
        """Select reasoning strategy based on confidence dynamics."""
        if dynamics.trajectory_type == "progressing":
            return Strategy.LINEAR
        elif dynamics.trajectory_type == "stagnant":
            return Strategy.ANALOGICAL
        elif dynamics.trajectory_type == "oscillating":
            return Strategy.DECOMPOSITION
        elif dynamics.trajectory_type == "declining":
            return Strategy.VERIFICATION
        return Strategy.LINEAR
    
    def retrieve_analogical_example(
        self,
        problem: str,
        similarity_threshold: float = 0.75
    ) -> Tuple[Optional[str], float]:
        """Retrieve similar problem for analogical transfer."""
        if self.retrieval_index is None:
            return None, 0.0
        
        # This would use the retrieval index to find similar problems
        # For now, return None (will be implemented with sentence embeddings)
        return None, 0.0
    
    def build_strategy_prompt(
        self,
        problem: str,
        current_strategy: Strategy,
        reasoning_history: List[ReasoningStep],
        dynamics: ConfidenceDynamics
    ) -> str:
        """Build prompt for current strategy."""
        base_prompt = self.prompts[current_strategy]
        
        # Include reasoning history summary
        history_summary = ""
        if reasoning_history:
            key_results = []
            for step in reasoning_history[-3:]:  # Last 3 steps
                # Extract any numerical results
                numbers = re.findall(r'-?\d+(?:\.\d+)?', step.text)
                if numbers:
                    key_results.append(numbers[-1])
            if key_results:
                history_summary = f"\n\nKey results so far: {', '.join(key_results)}"
        
        # Add strategy-specific instructions
        if current_strategy == Strategy.ANALOGICAL:
            example, similarity = self.retrieve_analogical_example(problem)
            if example and similarity > 0.5:
                base_prompt += f"\n\nSimilar problem: {example}"
                if similarity <= 0.75:
                    base_prompt += "\nPlease verify this analogy applies to the current problem."
            else:
                # Fallback to decomposition if no good analogy found
                base_prompt = self.prompts[Strategy.DECOMPOSITION]
        
        elif current_strategy == Strategy.DECOMPOSITION:
            base_prompt += "\n\nBreak this down into: (1) What do we know? (2) What do we need to find? (3) What steps connect them?"
        
        elif current_strategy == Strategy.VERIFICATION:
            base_prompt += "\n\nReview each step for: calculation errors, assumption validity, and logical consistency."
        
        full_prompt = f"{base_prompt}{history_summary}\n\nProblem: {problem}\n\n"
        return full_prompt
    
    def solve(
        self,
        problem: str,
        max_steps: int = 10,
        verbose: bool = False
    ) -> Dict:
        """Solve a problem using CDHR."""
        reasoning_history = []
        confidence_trajectory = []
        strategy_switches = 0
        current_strategy = Strategy.LINEAR
        
        for step_num in range(max_steps):
            # Build prompt for current step
            dynamics = self.compute_dynamics(confidence_trajectory)
            
            # Check if we should switch strategy
            new_strategy = self.select_strategy(dynamics)
            if new_strategy != current_strategy:
                if strategy_switches < self.max_switches:
                    current_strategy = new_strategy
                    strategy_switches += 1
            
            prompt = self.build_strategy_prompt(
                problem, current_strategy, reasoning_history, dynamics
            )
            
            # Generate next step
            response, logprobs = self.llm.generate_with_logprobs(
                prompt,
                temperature=0.0,
                max_tokens=256
            )
            
            # Compute confidence
            token_conf = self.compute_token_confidence(logprobs)
            # Only compute consistency if beta < 1.0
            consistency_conf = 0.0 if self.beta == 1.0 else self.compute_consistency_confidence(
                prompt, response, k=3
            )
            composite_conf = self.compute_composite_confidence(token_conf, consistency_conf)
            
            # Record step
            step = ReasoningStep(
                text=response,
                token_confidence=token_conf,
                consistency_confidence=consistency_conf,
                composite_confidence=composite_conf,
                strategy=current_strategy
            )
            reasoning_history.append(step)
            confidence_trajectory.append(composite_conf)
            
            if verbose:
                print(f"Step {step_num+1}: {current_strategy.value}, conf={composite_conf:.3f}")
            
            # Check for completion
            if self._is_complete(response):
                break
        
        # Compile final answer
        full_reasoning = "\n".join([s.text for s in reasoning_history])
        final_answer = self._extract_answer(full_reasoning)
        
        # Strategy distribution
        strategy_counts = {}
        for s in reasoning_history:
            strategy_counts[s.strategy.value] = strategy_counts.get(s.strategy.value, 0) + 1
        
        return {
            "answer": final_answer,
            "reasoning": full_reasoning,
            "steps": len(reasoning_history),
            "strategy_switches": strategy_switches,
            "strategy_distribution": strategy_counts,
            "confidence_trajectory": confidence_trajectory,
            "final_confidence": confidence_trajectory[-1] if confidence_trajectory else 0.0,
        }
    
    def _is_complete(self, text: str) -> bool:
        """Check if reasoning appears complete."""
        completion_markers = [
            "the answer is",
            "therefore,",
            "in conclusion",
            "####",
            "final answer",
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in completion_markers)
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from reasoning text."""
        # Try various patterns
        patterns = [
            r'####\s*(-?\d+(?:\.\d+)?)',
            r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)',
            r'(?:therefore|thus|so)\s*.{0,20}?(-?\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else None
    
    def _extract_quick_answer(self, text: str) -> str:
        """Quick answer extraction for consistency check."""
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else text.strip()[:50]


class SimpleLLMEngine:
    """Simple wrapper for vLLM or similar inference engines."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        """Generate text from prompt."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_with_logprobs(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        max_tokens: int = 256
    ) -> Tuple[str, List[float]]:
        """Generate text and return log probabilities."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,  # Request logprobs
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # Extract logprobs
        logprobs = []
        if hasattr(outputs[0].outputs[0], 'logprobs') and outputs[0].outputs[0].logprobs:
            for logprob_dict in outputs[0].outputs[0].logprobs:
                if logprob_dict:
                    # Get the logprob of the generated token
                    token_id = list(logprob_dict.keys())[0]
                    logprobs.append(logprob_dict[token_id])
        
        return output_text, logprobs
