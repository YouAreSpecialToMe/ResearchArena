import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .utils import ROOT, ensure_dir, sentence_split


QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MNLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class Generator:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def _prompt(self, question: str, context: str = "") -> str:
        if context:
            return (
                "Answer with a short factual phrase only.\n"
                f"Question: {question}\n"
                f"Context: {context}\n"
                "Answer:"
            )
        return f"Answer with a short factual phrase only.\nQuestion: {question}\nAnswer:"

    @torch.inference_mode()
    def generate_batch(self, questions: List[str], contexts: List[str]) -> List[Dict]:
        prompts = [self._prompt(q, c) for q, c in zip(questions, contexts)]
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        start = time.time()
        outputs = self.model.generate(
            **batch,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.time() - start
        gen_ids = outputs.sequences[:, batch["input_ids"].shape[1] :]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        logprob_steps = [torch.log_softmax(score.float(), dim=-1) for score in outputs.scores]
        prob_steps = [torch.softmax(score.float(), dim=-1) for score in outputs.scores]
        rows = []
        for i, text in enumerate(texts):
            token_ids = gen_ids[i].tolist()
            chosen = []
            entropies = []
            for step, tok in enumerate(token_ids[: len(logprob_steps)]):
                if tok in {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}:
                    break
                chosen.append(float(logprob_steps[step][i, tok].item()))
                probs = prob_steps[step][i]
                entropies.append(float((-(probs * torch.log(probs + 1e-12))).sum().item()))
            rows.append(
                {
                    "answer": text.strip(),
                    "mean_logprob": float(np.mean(chosen)) if chosen else -10.0,
                    "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
                    "latency_sec": elapsed / max(len(texts), 1),
                    "prompt_length": int(batch["attention_mask"][i].sum().item()),
                }
            )
        return rows


class FeatureModels:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = None
        self.mnli_tokenizer = None
        self.mnli_model = None
        self._load()

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(MINILM_MODEL, device=self.device)
        self.mnli_tokenizer = AutoTokenizer.from_pretrained(MNLI_MODEL)
        self.mnli_model = AutoModelForSequenceClassification.from_pretrained(MNLI_MODEL)
        self.mnli_model.to(self.device)
        self.mnli_model.eval()

    @torch.inference_mode()
    def max_similarity(self, answer: str, context: str) -> float:
        sents = sentence_split(context)[:24] or [context[:256]]
        embeds = self.embedder.encode([answer] + sents, convert_to_tensor=True, normalize_embeddings=True)
        sims = torch.matmul(embeds[0], embeds[1:].T)
        return float(torch.max(sims).item()) if sims.numel() else 0.0

    @torch.inference_mode()
    def nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        toks = self.mnli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        logits = self.mnli_model(**toks).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        labels = {v.lower(): i for i, v in self.mnli_model.config.id2label.items()}
        entail = float(probs[labels.get("entailment", 2)])
        contra = float(probs[labels.get("contradiction", 0)])
        neutral = float(probs[labels.get("neutral", 1)])
        return {"entailment": entail, "contradiction": contra, "neutral": neutral}
