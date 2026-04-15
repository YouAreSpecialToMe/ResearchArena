import math
import numpy as np
import torch
from .evaluation import normalize_answer, token_f1 as compute_token_f1


def pra_em(a_param, a_rag):
    """Exact match between parametric and RAG answers."""
    return int(normalize_answer(a_param) == normalize_answer(a_rag))


def pra_f1(a_param, a_rag):
    """Token F1 between parametric and RAG answers."""
    return compute_token_f1(a_param, [a_rag])


def pra_nli_batch(premises, hypotheses, nli_model, nli_tokenizer, device='cuda', batch_size=64):
    """Batch NLI inference, return P(entailment) for each pair."""
    scores = []
    nli_model.eval()
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hypotheses[i:i+batch_size]
        inputs = nli_tokenizer(
            batch_p, batch_h,
            return_tensors='pt', padding=True, truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # cross-encoder/nli-deberta-v3-large: 0=contradiction, 1=entailment, 2=neutral
            entailment_probs = probs[:, 1].cpu().numpy().tolist()
            scores.extend(entailment_probs)
    return scores


def pra_tpd(parametric_logprob, rag_logprob, a_param, a_rag, alpha=0.5):
    """Combined token probability delta + output agreement."""
    f1 = pra_f1(a_param, a_rag)
    delta = rag_logprob - parametric_logprob
    sigmoid_delta = 1.0 / (1.0 + math.exp(-delta))
    return alpha * f1 + (1 - alpha) * sigmoid_delta
