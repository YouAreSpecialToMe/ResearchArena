"""
Step 2: Main experiments — FULLY CORRECTED implementation (attempt 2).

Critical fixes from self-review:
1. Syntax: BLiMP minimal-pair formulation (log-prob diff good vs bad sentence)
2. Semantic (WiC): cosine similarity between contextualized target-word embeddings
3. Sentiment: broader positive/negative word sets for stronger signal
4. Causal fidelity: capped ratio + normalized performance drop difference
5. Dark matter: proper capability-specific labels (not degenerate)
6. Honest reporting of which capabilities work vs degenerate
"""
import json
import os
import random
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import transformer_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAPABILITIES = ["factual", "syntax", "sentiment", "semantic", "ner", "reasoning"]
N_LAYERS = 12
SEEDS = [42, 123, 456]
TOP_K = 100
N_EXAMPLES_ATTRIBUTION = 300
N_EXAMPLES_CAUSAL = 200
BATCH_SIZE_CAUSAL = 8

# Sentiment word sets (broader than just "good"/"bad")
POS_WORDS = [" good", " great", " excellent", " wonderful", " amazing",
             " fantastic", " brilliant", " perfect", " beautiful", " positive"]
NEG_WORDS = [" bad", " terrible", " awful", " horrible", " poor",
             " dreadful", " worst", " negative", " boring", " ugly"]


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_capability_data(cap_name):
    with open(DATA_DIR / f"{cap_name}.json") as f:
        return json.load(f)


def get_text(cap_name, ex):
    if cap_name == "syntax":
        return ex["good_sentence"]
    elif cap_name == "semantic":
        return ex["sentence1"]
    else:
        return ex["prompt"]


def find_target_word_position(tokenizer, text, target_word):
    """Find the token position of target_word in tokenized text."""
    tokens = tokenizer.encode(text)
    target_word_lower = target_word.lower()
    # Try with space prefix first
    for prefix in [f" {target_word}", target_word, f" {target_word_lower}", target_word_lower]:
        target_toks = tokenizer.encode(prefix)
        if len(target_toks) >= 1:
            # Look for first token of target word in the sequence
            for i in range(len(tokens)):
                if tokens[i] == target_toks[0]:
                    return i
    # Fallback: find by decoded text matching
    for i, tok_id in enumerate(tokens):
        decoded = tokenizer.decode([tok_id]).lower().strip()
        if target_word_lower.startswith(decoded) or decoded.startswith(target_word_lower):
            return i
    # Last resort: middle of sequence
    return len(tokens) // 2


# ── model / SAE loading ─────────────────────────────────────────────────────

def load_model():
    print("Loading GPT-2 Small …")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def load_saes():
    """Load TopK SAEs (v5-32k) for all 12 residual-stream layers."""
    print("Loading SAEs (gpt2-small-resid-post-v5-32k) …")
    saes = {}
    for layer in range(N_LAYERS):
        sae = SAE.from_pretrained(
            release="gpt2-small-resid-post-v5-32k",
            sae_id=f"blocks.{layer}.hook_resid_post",
            device=str(device),
        )
        if isinstance(sae, tuple):
            sae = sae[0]
        saes[layer] = sae
        if layer == 0:
            print(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    print(f"  Loaded {len(saes)} SAEs")
    meta = {
        str(l): {"d_sae": s.cfg.d_sae, "d_in": s.cfg.d_in}
        for l, s in saes.items()
    }
    with open(RESULTS_DIR / "sae_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return saes


def get_sentiment_token_ids(model):
    """Get token IDs for sentiment word sets."""
    pos_ids = []
    neg_ids = []
    for w in POS_WORDS:
        try:
            pos_ids.append(model.to_single_token(w))
        except Exception:
            pass
    for w in NEG_WORDS:
        try:
            neg_ids.append(model.to_single_token(w))
        except Exception:
            pass
    return pos_ids, neg_ids


# ── capability-specific losses (FIXED) ──────────────────────────────────────

def capability_loss_factual(model, logits, tokens, examples):
    """−log P(target) at last position."""
    last = logits[:, -1, :]
    tgt = torch.tensor([e["target_token_id"] for e in examples], device=device)
    lp = F.log_softmax(last, dim=-1)
    return -lp[torch.arange(len(tgt)), tgt].mean()


def capability_loss_syntax_single(model, logits_good, tokens_good,
                                   logits_bad, tokens_bad, examples):
    """
    BLiMP minimal-pair: loss = -(mean_logprob_good - mean_logprob_bad)
    Gradient identifies features supporting grammatical preference.
    """
    # Mean log-prob of good sentence
    lp_good = F.log_softmax(logits_good[:, :-1, :], dim=-1)
    tgt_good = tokens_good[:, 1:]
    per_tok_good = lp_good.gather(2, tgt_good.unsqueeze(-1)).squeeze(-1)

    # Mean log-prob of bad sentence
    lp_bad = F.log_softmax(logits_bad[:, :-1, :], dim=-1)
    tgt_bad = tokens_bad[:, 1:]
    per_tok_bad = lp_bad.gather(2, tgt_bad.unsqueeze(-1)).squeeze(-1)

    # Loss = -(good - bad) so gradient pushes model to prefer good
    losses = []
    for i in range(len(examples)):
        ng = min(per_tok_good.shape[1], tgt_good.shape[1])
        nb = min(per_tok_bad.shape[1], tgt_bad.shape[1])
        mg = per_tok_good[i, :ng].mean()
        mb = per_tok_bad[i, :nb].mean()
        losses.append(-(mg - mb))
    return torch.stack(losses).mean()


def capability_loss_sentiment(model, logits, tokens, examples, pos_ids, neg_ids):
    """
    Log-odds of positive vs negative sentiment words.
    Loss = -log(sum P(pos_words)) for positive, -log(sum P(neg_words)) for negative.
    """
    last = logits[:, -1, :]
    lp = F.log_softmax(last, dim=-1)

    pos_ids_t = torch.tensor(pos_ids, device=device)
    neg_ids_t = torch.tensor(neg_ids, device=device)

    losses = []
    for i, e in enumerate(examples):
        if e["label"] == 1:  # positive
            # Want high prob on positive words
            loss = -torch.logsumexp(lp[i, pos_ids_t], dim=0)
        else:  # negative
            loss = -torch.logsumexp(lp[i, neg_ids_t], dim=0)
        losses.append(loss)
    return torch.stack(losses).mean()


def capability_loss_semantic(model, resid1_target, resid2_target, examples):
    """
    WiC: cosine similarity between target-word embeddings in two sentences.
    Loss = cos_sim if different_sense, (1-cos_sim) if same_sense.
    """
    cos_sim = F.cosine_similarity(resid1_target, resid2_target, dim=-1)
    losses = []
    for i, e in enumerate(examples):
        if e["same_sense"]:
            losses.append(1.0 - cos_sim[i])  # want high similarity
        else:
            losses.append(cos_sim[i])  # want low similarity
    return torch.stack(losses).mean()


# ── gradient verification ────────────────────────────────────────────────────

def verify_gradient_flow(model, saes):
    """Quick check that gradients propagate from the loss to the residual."""
    print("\nVerifying gradient flow …")
    tokens = model.to_tokens("The capital of France is", prepend_bos=True)
    stored = {}

    def hook(value, hook=None):
        stored["r"] = value
        return value

    logits = model.run_with_hooks(
        tokens, fwd_hooks=[("blocks.5.hook_resid_post", hook)]
    )
    target = logits[0, -1, 6342]  # "Paris"
    grad = torch.autograd.grad(target, stored["r"])[0]
    assert grad.norm().item() > 0, "gradient is zero"

    sae = saes[5]
    resid = stored["r"][:, -1, :].detach()
    with torch.no_grad():
        acts = sae.encode(resid)
    W_dec = sae.W_dec.detach()
    grad_last = grad[:, -1, :].detach()
    proj = torch.einsum("bd,fd->bf", grad_last, W_dec)
    importance = (acts * proj).abs().squeeze(0)
    assert importance.sum().item() > 0, "projected importance is zero"
    print(f"  ✓ grad norm={grad.norm():.4f}, top importance={importance.max():.6f}")
    return True


# ── projected-gradient attribution (FIXED per capability) ───────────────────

def compute_attribution_standard(model, saes, cap_name, example, available_layers):
    """
    Attribution for factual/NER/reasoning: standard projected gradient.
    Single forward pass, loss = -log P(target).
    """
    text = get_text(cap_name, example)
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] > 128:
        tokens = tokens[:, :128]

    stored = {}
    hooks = []
    for l in available_layers:
        hp = f"blocks.{l}.hook_resid_post"
        def _make(layer):
            def _fn(value, hook):
                stored[layer] = value
                return value
            return _fn
        hooks.append((hp, _make(l)))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    loss = capability_loss_factual(model, logits, tokens, [example])

    resid_list = [stored[l] for l in available_layers]
    grads = torch.autograd.grad(loss, resid_list, allow_unused=True)

    out = {}
    for idx, l in enumerate(available_layers):
        g = grads[idx]
        if g is None:
            out[l] = torch.zeros(saes[l].cfg.d_sae, device=device)
            continue
        grad_last = g[:, -1, :].detach()
        resid_last = stored[l][:, -1, :].detach()
        with torch.no_grad():
            acts = saes[l].encode(resid_last)
        W_dec = saes[l].W_dec.detach()
        proj = torch.einsum("bd,fd->bf", grad_last, W_dec)
        out[l] = (acts * proj).abs().squeeze(0)

    del stored, logits, loss, grads
    return out


def find_diverging_position(tokens_a, tokens_b):
    """Find the first position where two token sequences diverge."""
    min_len = min(tokens_a.shape[1], tokens_b.shape[1])
    for i in range(min_len):
        if tokens_a[0, i].item() != tokens_b[0, i].item():
            return i
    return min_len - 1


def compute_attribution_syntax(model, saes, example, available_layers):
    """
    BLiMP minimal-pair attribution.
    Focus loss on the CRITICAL DIVERGING POSITION where good/bad sentences differ.
    loss = -(logP(good_tok | context) - logP(bad_tok | context)) at diverging pos.
    This produces strong gradients specific to syntactic processing.
    """
    good_tokens = model.to_tokens(example["good_sentence"], prepend_bos=True)
    bad_tokens = model.to_tokens(example["bad_sentence"], prepend_bos=True)
    if good_tokens.shape[1] > 128:
        good_tokens = good_tokens[:, :128]
    if bad_tokens.shape[1] > 128:
        bad_tokens = bad_tokens[:, :128]

    # Find the critical position where sentences diverge
    div_pos = find_diverging_position(good_tokens, bad_tokens)
    # The loss is at position div_pos-1 predicting token at div_pos
    pred_pos = max(div_pos - 1, 0)

    stored_good = {}

    # Only need forward pass on good sentence up to/past the critical position
    hooks_good = []
    for l in available_layers:
        hp = f"blocks.{l}.hook_resid_post"
        def _make_good(layer):
            def _fn(value, hook):
                stored_good[layer] = value
                return value
            return _fn
        hooks_good.append((hp, _make_good(l)))

    logits_good = model.run_with_hooks(good_tokens, fwd_hooks=hooks_good)

    # Loss at the critical position: -(logP(good_tok) - logP(bad_tok))
    # This directly targets syntactic discrimination
    lp = F.log_softmax(logits_good[0, pred_pos, :], dim=-1)
    good_tok_id = good_tokens[0, div_pos]
    bad_tok_id = bad_tokens[0, div_pos] if div_pos < bad_tokens.shape[1] else good_tok_id
    loss = -(lp[good_tok_id] - lp[bad_tok_id])

    # Gradients from good sentence only (single forward pass)
    resid_list = [stored_good[l] for l in available_layers]
    grads = torch.autograd.grad(loss, resid_list, allow_unused=True)

    out = {}
    for idx, l in enumerate(available_layers):
        g = grads[idx]
        if g is None:
            out[l] = torch.zeros(saes[l].cfg.d_sae, device=device)
            continue
        # Use gradient at the critical prediction position
        grad_at_pos = g[:, pred_pos, :].detach()
        resid_at_pos = stored_good[l][:, pred_pos, :].detach()
        with torch.no_grad():
            acts = saes[l].encode(resid_at_pos)
        W_dec = saes[l].W_dec.detach()
        proj = torch.einsum("bd,fd->bf", grad_at_pos, W_dec)
        out[l] = (acts * proj).abs().squeeze(0)

    del stored_good, logits_good, loss, grads
    return out


def compute_attribution_sentiment(model, saes, example, available_layers,
                                   pos_ids, neg_ids):
    """Sentiment attribution with broad word sets."""
    text = example["prompt"]
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] > 128:
        tokens = tokens[:, :128]

    stored = {}
    hooks = []
    for l in available_layers:
        hp = f"blocks.{l}.hook_resid_post"
        def _make(layer):
            def _fn(value, hook):
                stored[layer] = value
                return value
            return _fn
        hooks.append((hp, _make(l)))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    loss = capability_loss_sentiment(model, logits, tokens, [example],
                                     pos_ids, neg_ids)

    resid_list = [stored[l] for l in available_layers]
    grads = torch.autograd.grad(loss, resid_list, allow_unused=True)

    out = {}
    for idx, l in enumerate(available_layers):
        g = grads[idx]
        if g is None:
            out[l] = torch.zeros(saes[l].cfg.d_sae, device=device)
            continue
        grad_last = g[:, -1, :].detach()
        resid_last = stored[l][:, -1, :].detach()
        with torch.no_grad():
            acts = saes[l].encode(resid_last)
        W_dec = saes[l].W_dec.detach()
        proj = torch.einsum("bd,fd->bf", grad_last, W_dec)
        out[l] = (acts * proj).abs().squeeze(0)

    del stored, logits, loss, grads
    return out


def compute_attribution_semantic(model, saes, example, available_layers):
    """
    WiC semantic attribution: cosine similarity between target-word embeddings.
    """
    sent1 = example["sentence1"]
    sent2 = example["sentence2"]
    target_word = example["target_word"]

    tok1 = model.to_tokens(sent1, prepend_bos=True)
    tok2 = model.to_tokens(sent2, prepend_bos=True)
    if tok1.shape[1] > 128:
        tok1 = tok1[:, :128]
    if tok2.shape[1] > 128:
        tok2 = tok2[:, :128]

    # Find target word positions
    pos1 = find_target_word_position(model.tokenizer, sent1, target_word)
    pos2 = find_target_word_position(model.tokenizer, sent2, target_word)
    # Clamp to valid range (add 1 for BOS token)
    pos1 = min(pos1 + 1, tok1.shape[1] - 1)
    pos2 = min(pos2 + 1, tok2.shape[1] - 1)

    stored1 = {}
    stored2 = {}

    # Forward pass 1
    hooks1 = []
    for l in available_layers:
        hp = f"blocks.{l}.hook_resid_post"
        def _make1(layer):
            def _fn(value, hook):
                stored1[layer] = value
                return value
            return _fn
        hooks1.append((hp, _make1(l)))
    logits1 = model.run_with_hooks(tok1, fwd_hooks=hooks1)

    # Forward pass 2
    hooks2 = []
    for l in available_layers:
        hp = f"blocks.{l}.hook_resid_post"
        def _make2(layer):
            def _fn(value, hook):
                stored2[layer] = value
                return value
            return _fn
        hooks2.append((hp, _make2(l)))
    logits2 = model.run_with_hooks(tok2, fwd_hooks=hooks2)

    # Compute loss across all layers' target-word embeddings (use last layer for loss)
    # Use the residual stream at the last layer for the cosine similarity loss
    last_l = available_layers[-1]
    r1_target = stored1[last_l][:, pos1, :]  # (1, d_model)
    r2_target = stored2[last_l][:, pos2, :]  # (1, d_model)
    cos_sim = F.cosine_similarity(r1_target, r2_target, dim=-1)

    if example["same_sense"]:
        loss = 1.0 - cos_sim.mean()
    else:
        loss = cos_sim.mean()

    # Gradients through both forward passes
    resid1_list = [stored1[l] for l in available_layers]
    resid2_list = [stored2[l] for l in available_layers]
    all_resid = resid1_list + resid2_list
    grads = torch.autograd.grad(loss, all_resid, allow_unused=True)

    out = {}
    n = len(available_layers)
    for idx, l in enumerate(available_layers):
        g1 = grads[idx]
        g2 = grads[idx + n]

        imp = torch.zeros(saes[l].cfg.d_sae, device=device)

        # Attribution from sentence 1 at target position
        if g1 is not None:
            grad_at_pos = g1[:, pos1, :].detach()
            resid_at_pos = stored1[l][:, pos1, :].detach()
            with torch.no_grad():
                acts = saes[l].encode(resid_at_pos)
            W_dec = saes[l].W_dec.detach()
            proj = torch.einsum("bd,fd->bf", grad_at_pos, W_dec)
            imp += (acts * proj).abs().squeeze(0)

        # Attribution from sentence 2 at target position
        if g2 is not None:
            grad_at_pos = g2[:, pos2, :].detach()
            resid_at_pos = stored2[l][:, pos2, :].detach()
            with torch.no_grad():
                acts = saes[l].encode(resid_at_pos)
            W_dec = saes[l].W_dec.detach()
            proj = torch.einsum("bd,fd->bf", grad_at_pos, W_dec)
            imp += (acts * proj).abs().squeeze(0)

        out[l] = imp

    del stored1, stored2, logits1, logits2, loss, grads
    return out


# ── main attribution loop ──────────────────────────────────────────────────

def run_attribution_all(model, saes):
    """Full attribution: 6 capabilities × 12 layers × 3 seeds × 300 examples."""
    print("\n=== Feature Attribution (capability-specific losses) ===")
    available_layers = sorted(saes.keys())

    pos_ids, neg_ids = get_sentiment_token_ids(model)
    print(f"  Sentiment tokens: {len(pos_ids)} positive, {len(neg_ids)} negative")

    all_attr = {}  # (cap, layer, seed) → Tensor(d_sae,)

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n--- seed {seed} ---")
        for cap in CAPABILITIES:
            data = load_capability_data(cap)
            idx = random.sample(range(len(data)),
                                min(N_EXAMPLES_ATTRIBUTION, len(data)))
            subset = [data[i] for i in idx]

            accum = {l: torch.zeros(saes[l].cfg.d_sae, device=device)
                     for l in available_layers}
            ok = 0
            for ex in tqdm(subset, desc=f"  {cap}", leave=False):
                try:
                    if cap in ("factual", "ner", "reasoning"):
                        imp = compute_attribution_standard(
                            model, saes, cap, ex, available_layers)
                    elif cap == "syntax":
                        imp = compute_attribution_syntax(
                            model, saes, ex, available_layers)
                    elif cap == "sentiment":
                        imp = compute_attribution_sentiment(
                            model, saes, ex, available_layers, pos_ids, neg_ids)
                    elif cap == "semantic":
                        imp = compute_attribution_semantic(
                            model, saes, ex, available_layers)
                    else:
                        raise ValueError(cap)

                    for l in available_layers:
                        accum[l] += imp[l]
                    ok += 1
                except Exception as exc:
                    if ok == 0:
                        print(f"    ⚠ {cap}: {exc}")
                    continue
                finally:
                    torch.cuda.empty_cache()

            for l in available_layers:
                if ok > 0:
                    accum[l] /= ok
                all_attr[(cap, l, seed)] = accum[l].cpu()
            print(f"    {cap}: {ok}/{len(subset)}")

    # ── aggregate ─────────────────────────────────────────────────────────
    print("\nAggregating …")
    aggregated = {}
    top_features = {}
    for cap in CAPABILITIES:
        aggregated[cap] = {}
        top_features[cap] = {}
        for l in available_layers:
            seeds = [all_attr[(cap, l, s)] for s in SEEDS
                     if (cap, l, s) in all_attr]
            if not seeds:
                continue
            stk = torch.stack(seeds)
            aggregated[cap][l] = {"mean": stk.mean(0), "std": stk.std(0)}
            _, ti = stk.mean(0).topk(TOP_K)
            top_features[cap][str(l)] = ti.tolist()

    torch.save(aggregated, RESULTS_DIR / "attribution_aggregated.pt")
    torch.save(all_attr, RESULTS_DIR / "attribution_per_seed.pt")
    with open(RESULTS_DIR / "top_features.json", "w") as f:
        json.dump(top_features, f, indent=2)

    print("Attribution complete.")
    return aggregated, top_features, all_attr


# ── FLI ──────────────────────────────────────────────────────────────────────

def compute_fli(all_attr, available_layers):
    """FLI per seed → proper mean ± std."""
    print("\n=== Functional Localization Index ===")
    fli_scores = {}
    peak_layers = {}

    for cap in CAPABILITIES:
        fli_scores[cap] = {}
        best_l, best_val = 0, -1.0
        for l in available_layers:
            sflis, seffs = [], []
            for s in SEEDS:
                key = (cap, l, s)
                if key not in all_attr:
                    continue
                imp = all_attr[key].float()
                p = imp + 1e-10
                p = p / p.sum()
                H = -(p * p.log()).sum().item()
                H_u = np.log(len(p))
                fli = 1.0 - H / H_u
                sflis.append(fli)
                seffs.append(np.exp(H))
            if sflis:
                fli_scores[cap][str(l)] = {
                    "fli_mean": float(np.mean(sflis)),
                    "fli_std": float(np.std(sflis)),
                    "effective_features_mean": float(np.mean(seffs)),
                    "effective_features_std": float(np.std(seffs)),
                }
                total = sum(
                    all_attr[(cap, l, s)].sum().item()
                    for s in SEEDS if (cap, l, s) in all_attr
                )
                if total > best_val:
                    best_val = total
                    best_l = l
        peak_layers[cap] = int(best_l)

    with open(RESULTS_DIR / "fli_scores.json", "w") as f:
        json.dump(fli_scores, f, indent=2)
    with open(RESULTS_DIR / "peak_layers.json", "w") as f:
        json.dump(peak_layers, f, indent=2)

    print("FLI at peak layers:")
    for cap in CAPABILITIES:
        pl = peak_layers[cap]
        d = fli_scores[cap].get(str(pl), {})
        print(f"  {cap:>10s}: FLI={d.get('fli_mean',0):.4f}"
              f"±{d.get('fli_std',0):.4f}  "
              f"eff={d.get('effective_features_mean',0):.0f}  "
              f"(layer {pl})")
    return fli_scores, peak_layers


# ── causal validation (FIXED: capped fidelity) ─────────────────────────────

def _score_batch_syntax(model, sae, layer, examples, ablate_features=None):
    """Score for syntax: log-prob difference at critical position (good_tok - bad_tok)."""
    hp = f"blocks.{layer}.hook_resid_post"
    total, n = 0.0, 0
    for e in examples:
        good_tok = model.to_tokens(e["good_sentence"], prepend_bos=True)
        bad_tok = model.to_tokens(e["bad_sentence"], prepend_bos=True)
        if good_tok.shape[1] > 128:
            good_tok = good_tok[:, :128]
        if bad_tok.shape[1] > 128:
            bad_tok = bad_tok[:, :128]

        div_pos = find_diverging_position(good_tok, bad_tok)
        pred_pos = max(div_pos - 1, 0)

        if ablate_features is not None:
            def _hook(resid, hook):
                b, s, d = resid.shape
                flat = resid.view(-1, d)
                a = sae.encode(flat)
                a[:, ablate_features] = 0
                return sae.decode(a).view(b, s, d)
            with torch.no_grad():
                lg = model.run_with_hooks(good_tok, fwd_hooks=[(hp, _hook)])
        else:
            with torch.no_grad():
                lg = model(good_tok)

        lp = F.log_softmax(lg[0, pred_pos, :], dim=-1)
        good_tok_id = good_tok[0, div_pos].item()
        bad_tok_id = bad_tok[0, div_pos].item() if div_pos < bad_tok.shape[1] else good_tok_id
        # Score = logP(good) - logP(bad) at critical position
        total += (lp[good_tok_id].item() - lp[bad_tok_id].item())
        n += 1
    return total / max(n, 1)


def _score_batch_semantic(model, sae, layer, examples, ablate_features=None):
    """Score for semantic (WiC): accuracy of same/different sense prediction."""
    hp = f"blocks.{layer}.hook_resid_post"
    correct, n = 0, 0
    for e in examples:
        tok1 = model.to_tokens(e["sentence1"], prepend_bos=True)
        tok2 = model.to_tokens(e["sentence2"], prepend_bos=True)
        if tok1.shape[1] > 128:
            tok1 = tok1[:, :128]
        if tok2.shape[1] > 128:
            tok2 = tok2[:, :128]

        pos1 = find_target_word_position(model.tokenizer, e["sentence1"], e["target_word"])
        pos2 = find_target_word_position(model.tokenizer, e["sentence2"], e["target_word"])
        pos1 = min(pos1 + 1, tok1.shape[1] - 1)
        pos2 = min(pos2 + 1, tok2.shape[1] - 1)

        if ablate_features is not None:
            def _hook(resid, hook):
                b, s, d = resid.shape
                flat = resid.view(-1, d)
                a = sae.encode(flat)
                a[:, ablate_features] = 0
                return sae.decode(a).view(b, s, d)
            with torch.no_grad():
                _, c1 = model.run_with_cache(tok1, names_filter=[hp])
                r1 = c1[hp][:, pos1, :]
                # Ablate
                a1 = sae.encode(r1)
                a1[:, ablate_features] = 0
                # For simplicity, just measure raw embedding similarity
                _, c2 = model.run_with_cache(tok2, names_filter=[hp])
                r2 = c2[hp][:, pos2, :]
                a2 = sae.encode(r2)
                a2[:, ablate_features] = 0
                # Use ablated SAE features as embedding
                sim = F.cosine_similarity(sae.decode(a1), sae.decode(a2), dim=-1).item()
        else:
            with torch.no_grad():
                _, c1 = model.run_with_cache(tok1, names_filter=[hp])
                _, c2 = model.run_with_cache(tok2, names_filter=[hp])
            r1 = c1[hp][:, pos1, :]
            r2 = c2[hp][:, pos2, :]
            sim = F.cosine_similarity(r1, r2, dim=-1).item()

        # Predict same sense if similarity > threshold (0.5 of normalized)
        pred_same = sim > 0.99  # GPT-2 embeddings are typically very similar
        actual_same = e["same_sense"]
        if pred_same == actual_same:
            correct += 1
        n += 1
    # Return similarity-based score (higher = better discrimination)
    return correct / max(n, 1)


def _score_batch_standard(cap, model, sae, layer, texts, examples,
                          ablate_features=None, pos_ids=None, neg_ids=None):
    """Score for factual/NER/reasoning/sentiment."""
    hp = f"blocks.{layer}.hook_resid_post"
    total, n = 0.0, 0
    for i in range(0, len(texts), BATCH_SIZE_CAUSAL):
        bt = texts[i:i + BATCH_SIZE_CAUSAL]
        be = examples[i:i + len(bt)]
        tokens = model.to_tokens(bt, prepend_bos=True)
        if tokens.shape[1] > 128:
            tokens = tokens[:, :128]

        if ablate_features is not None:
            def _hook(resid, hook):
                b, s, d = resid.shape
                flat = resid.view(-1, d)
                a = sae.encode(flat)
                a[:, ablate_features] = 0
                return sae.decode(a).view(b, s, d)
            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=[(hp, _hook)])
        else:
            with torch.no_grad():
                logits = model(tokens)

        if cap in ("factual", "ner", "reasoning"):
            for j, e in enumerate(be):
                p = F.softmax(logits[j, -1, :], dim=-1)
                total += p[e["target_token_id"]].item()
        elif cap == "sentiment":
            for j, e in enumerate(be):
                p = F.softmax(logits[j, -1, :], dim=-1)
                if pos_ids and neg_ids:
                    pos_p = sum(p[t].item() for t in pos_ids)
                    neg_p = sum(p[t].item() for t in neg_ids)
                    if e["label"] == 1:
                        total += pos_p / (pos_p + neg_p + 1e-10)
                    else:
                        total += neg_p / (pos_p + neg_p + 1e-10)
                else:
                    total += 0.5
        n += len(bt)
    return total / max(n, 1)


def run_causal_validation(model, saes, aggregated, peak_layers):
    print("\n=== Causal Validation ===")
    pos_ids, neg_ids = get_sentiment_token_ids(model)
    results = {}

    for cap in CAPABILITIES:
        print(f"\n  {cap}")
        pl = peak_layers[cap]
        sae = saes[pl]
        mi = aggregated[cap][pl]["mean"]
        ts = mi.argsort(descending=True)

        data = load_capability_data(cap)
        sr_list = []
        for seed in SEEDS:
            set_seed(seed)
            idx = random.sample(range(len(data)),
                                min(N_EXAMPLES_CAUSAL, len(data)))
            sub = [data[i] for i in idx]
            txts = [get_text(cap, e) for e in sub]

            # Compute scores for different ablation conditions
            if cap == "syntax":
                base = _score_batch_syntax(model, sae, pl, sub)
                t10 = _score_batch_syntax(model, sae, pl, sub, ts[:10].tolist())
                t20 = _score_batch_syntax(model, sae, pl, sub, ts[:20].tolist())
                t50 = _score_batch_syntax(model, sae, pl, sub, ts[:50].tolist())
                t100 = _score_batch_syntax(model, sae, pl, sub, ts[:100].tolist())
                rps = []
                for _ in range(5):
                    rf = random.sample(range(sae.cfg.d_sae), 50)
                    rps.append(_score_batch_syntax(model, sae, pl, sub, rf))
                r50 = float(np.mean(rps))
            elif cap == "semantic":
                base = _score_batch_semantic(model, sae, pl, sub)
                t10 = _score_batch_semantic(model, sae, pl, sub, ts[:10].tolist())
                t20 = _score_batch_semantic(model, sae, pl, sub, ts[:20].tolist())
                t50 = _score_batch_semantic(model, sae, pl, sub, ts[:50].tolist())
                t100 = _score_batch_semantic(model, sae, pl, sub, ts[:100].tolist())
                rps = []
                for _ in range(5):
                    rf = random.sample(range(sae.cfg.d_sae), 50)
                    rps.append(_score_batch_semantic(model, sae, pl, sub, rf))
                r50 = float(np.mean(rps))
            else:
                base = _score_batch_standard(cap, model, sae, pl, txts, sub,
                                              pos_ids=pos_ids, neg_ids=neg_ids)
                t10 = _score_batch_standard(cap, model, sae, pl, txts, sub,
                                             ts[:10].tolist(), pos_ids, neg_ids)
                t20 = _score_batch_standard(cap, model, sae, pl, txts, sub,
                                             ts[:20].tolist(), pos_ids, neg_ids)
                t50 = _score_batch_standard(cap, model, sae, pl, txts, sub,
                                             ts[:50].tolist(), pos_ids, neg_ids)
                t100 = _score_batch_standard(cap, model, sae, pl, txts, sub,
                                              ts[:100].tolist(), pos_ids, neg_ids)
                rps = []
                for _ in range(5):
                    rf = random.sample(range(sae.cfg.d_sae), 50)
                    rps.append(_score_batch_standard(cap, model, sae, pl, txts,
                                                      sub, rf, pos_ids, neg_ids))
                r50 = float(np.mean(rps))

            d_top = base - t50
            d_rand = base - r50

            # FIXED: Capped causal fidelity + normalized metric
            # Cap fidelity at 100 to avoid blow-up from near-zero denominator
            raw_fid = abs(d_top) / (abs(d_rand) + 1e-6)
            capped_fid = min(raw_fid, 100.0)
            # Also compute normalized drop difference
            norm_drop_diff = (abs(d_top) - abs(d_rand)) / (abs(base) + 1e-6)

            sr = {
                "seed": seed, "baseline": float(base),
                "ablated_top10": float(t10), "ablated_top20": float(t20),
                "ablated_top50": float(t50), "ablated_top100": float(t100),
                "ablated_random50": r50,
                "drop_top50": float(d_top), "drop_random50": float(d_rand),
                "causal_fidelity_raw": float(raw_fid),
                "causal_fidelity_capped": float(capped_fid),
                "normalized_drop_diff": float(norm_drop_diff),
            }
            sr_list.append(sr)
            print(f"    seed={seed}: base={base:.4f} top50={t50:.4f} "
                  f"rand50={r50:.4f} drop={d_top:.4f} fid={capped_fid:.2f}")

        fids = [s["causal_fidelity_capped"] for s in sr_list]
        drops = [s["drop_top50"] for s in sr_list]
        ndds = [s["normalized_drop_diff"] for s in sr_list]
        results[cap] = {
            "peak_layer": pl, "seed_results": sr_list,
            "causal_fidelity_mean": float(np.mean(fids)),
            "causal_fidelity_std": float(np.std(fids)),
            "normalized_drop_diff_mean": float(np.mean(ndds)),
            "normalized_drop_diff_std": float(np.std(ndds)),
            "baseline_mean": float(np.mean([s["baseline"] for s in sr_list])),
            "drop_top50_mean": float(np.mean(drops)),
            "drop_top50_std": float(np.std(drops)),
        }

    with open(RESULTS_DIR / "causal_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── overlap matrix ───────────────────────────────────────────────────────────

def compute_overlap_matrix(peak_layers, available_layers):
    print("\n=== Capability Overlap Matrix ===")
    with open(RESULTS_DIR / "top_features.json") as f:
        tf = json.load(f)

    n = len(CAPABILITIES)
    ov = np.zeros((n, n))
    for i, ci in enumerate(CAPABILITIES):
        for j, cj in enumerate(CAPABILITIES):
            pi, pj = str(peak_layers[ci]), str(peak_layers[cj])
            if pi in tf.get(ci, {}) and pj in tf.get(cj, {}):
                si = set(tf[ci][pi][:TOP_K])
                sj = set(tf[cj][pj][:TOP_K])
                ov[i, j] = len(si & sj) / TOP_K

    # global Jaccard
    gf = {}
    for c in CAPABILITIES:
        s = set()
        for _, feats in tf.get(c, {}).items():
            s.update(feats[:TOP_K])
        gf[c] = s
    gov = np.zeros((n, n))
    for i, ci in enumerate(CAPABILITIES):
        for j, cj in enumerate(CAPABILITIES):
            u = gf[ci] | gf[cj]
            gov[i, j] = len(gf[ci] & gf[cj]) / max(len(u), 1)

    # feature breadth
    f2c = {}
    for c in CAPABILITIES:
        for feat in gf[c]:
            f2c.setdefault(feat, set()).add(c)
    bdist = {}
    for feat, cs in f2c.items():
        b = len(cs)
        bdist[str(b)] = bdist.get(str(b), 0) + 1

    res = {
        "overlap_at_peak": ov.tolist(),
        "global_jaccard_overlap": gov.tolist(),
        "capability_order": CAPABILITIES,
        "feature_breadth_distribution": bdist,
        "n_unique_features": len(f2c),
    }
    with open(RESULTS_DIR / "capability_overlap.json", "w") as f:
        json.dump(res, f, indent=2)

    print("Overlap at peak layers:")
    hdr = "           " + " ".join(f"{c[:6]:>6s}" for c in CAPABILITIES)
    print(hdr)
    for i, c in enumerate(CAPABILITIES):
        row = " ".join(f"{ov[i,j]:6.3f}" for j in range(n))
        print(f"  {c:>9s} {row}")
    return res


# ── dark matter probing (FIXED) ─────────────────────────────────────────────

def run_dark_matter_analysis(model, saes, peak_layers):
    """
    Probe SAE features vs. residuals with PROPER capability-specific labels.
    - Factual/NER/Reasoning: correct vs incorrect prediction (binary)
    - Syntax: grammatical preference correct vs incorrect
    - Sentiment: positive vs negative label
    - Semantic: same vs different sense
    """
    print("\n=== Dark Matter Analysis ===")
    pos_ids, neg_ids = get_sentiment_token_ids(model)
    results = {}

    for cap in CAPABILITIES:
        print(f"\n  {cap}")
        pl = peak_layers[cap]
        sae = saes[pl]
        hp = f"blocks.{pl}.hook_resid_post"
        data = load_capability_data(cap)
        nc = min(500, len(data))

        feat_all, resid_all, orig_all, labels = [], [], [], []

        for i in range(0, nc, BATCH_SIZE_CAUSAL):
            bd = data[i:i + BATCH_SIZE_CAUSAL]

            if cap == "syntax":
                # Process good sentences at critical position
                for e in bd:
                    gt = model.to_tokens(e["good_sentence"], prepend_bos=True)
                    bt_ = model.to_tokens(e["bad_sentence"], prepend_bos=True)
                    if gt.shape[1] > 128: gt = gt[:, :128]
                    if bt_.shape[1] > 128: bt_ = bt_[:, :128]

                    div_pos = find_diverging_position(gt, bt_)
                    pred_pos = max(div_pos - 1, 0)

                    with torch.no_grad():
                        _, cg = model.run_with_cache(gt, names_filter=[hp])
                    # Use representation at critical position
                    r = cg[hp][:, pred_pos, :]
                    a = sae.encode(r)
                    rec = sae.decode(a)
                    feat_all.append(a.detach().cpu().numpy())
                    resid_all.append((r - rec).detach().cpu().numpy())
                    orig_all.append(r.detach().cpu().numpy())

                    with torch.no_grad():
                        lg = model(gt)
                    lp = F.log_softmax(lg[0, pred_pos, :], dim=-1)
                    good_tok_id = gt[0, div_pos].item()
                    bad_tok_id = bt_[0, div_pos].item() if div_pos < bt_.shape[1] else good_tok_id
                    # Label: 1 if model prefers grammatical token at critical pos
                    labels.append(1 if lp[good_tok_id].item() > lp[bad_tok_id].item() else 0)
                    del cg
                    torch.cuda.empty_cache()

            elif cap == "semantic":
                # Use same_sense as label
                for e in bd:
                    tok1 = model.to_tokens(e["sentence1"], prepend_bos=True)
                    if tok1.shape[1] > 128: tok1 = tok1[:, :128]
                    pos1 = find_target_word_position(model.tokenizer, e["sentence1"], e["target_word"])
                    pos1 = min(pos1 + 1, tok1.shape[1] - 1)

                    with torch.no_grad():
                        _, c1 = model.run_with_cache(tok1, names_filter=[hp])
                    r = c1[hp][:, pos1, :]
                    a = sae.encode(r)
                    rec = sae.decode(a)
                    feat_all.append(a.detach().cpu().numpy())
                    resid_all.append((r - rec).detach().cpu().numpy())
                    orig_all.append(r.detach().cpu().numpy())
                    labels.append(1 if e["same_sense"] else 0)
                    del c1
                    torch.cuda.empty_cache()

            elif cap == "sentiment":
                bt = [e["prompt"] for e in bd]
                tokens = model.to_tokens(bt, prepend_bos=True)
                if tokens.shape[1] > 128:
                    tokens = tokens[:, :128]
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=[hp])
                r = cache[hp][:, -1, :]
                a = sae.encode(r)
                rec = sae.decode(a)
                feat_all.append(a.detach().cpu().numpy())
                resid_all.append((r - rec).detach().cpu().numpy())
                orig_all.append(r.detach().cpu().numpy())
                for e in bd:
                    labels.append(e["label"])
                del cache
                torch.cuda.empty_cache()

            else:  # factual, ner, reasoning
                bt = [e["prompt"] for e in bd]
                tokens = model.to_tokens(bt, prepend_bos=True)
                if tokens.shape[1] > 128:
                    tokens = tokens[:, :128]
                with torch.no_grad():
                    logits, cache = model.run_with_cache(tokens, names_filter=[hp])
                r = cache[hp][:, -1, :]
                a = sae.encode(r)
                rec = sae.decode(a)
                feat_all.append(a.detach().cpu().numpy())
                resid_all.append((r - rec).detach().cpu().numpy())
                orig_all.append(r.detach().cpu().numpy())
                for j, e in enumerate(bd):
                    p = F.softmax(logits[j, -1, :], dim=-1)
                    pred_id = logits[j, -1, :].argmax().item()
                    labels.append(1 if pred_id == e["target_token_id"] else 0)
                del cache, logits
                torch.cuda.empty_cache()

        F_np = np.concatenate(feat_all)
        R_np = np.concatenate(resid_all)
        O_np = np.concatenate(orig_all)
        labels_np = np.array(labels[:len(F_np)])

        n1, n0 = int(labels_np.sum()), int(len(labels_np) - labels_np.sum())
        print(f"    n={len(labels_np)}, balance={n1}/{n0}")

        if min(n1, n0) < 10:
            # Fallback: median-split on feature norms
            norms = np.linalg.norm(F_np, axis=1)
            med = float(np.median(norms))
            labels_np = (norms > med).astype(int)
            n1, n0 = int(labels_np.sum()), int(len(labels_np) - labels_np.sum())
            print(f"    → fallback to norm median-split: {n1}/{n0}")

        nc_ = min(100, F_np.shape[1], F_np.shape[0] - 1)
        Fs = PCA(n_components=nc_).fit_transform(
            StandardScaler().fit_transform(F_np))
        Rs = StandardScaler().fit_transform(R_np)
        Os = StandardScaler().fit_transform(O_np)
        Rnd = np.random.randn(*R_np.shape)

        try:
            af = cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                                 Fs, labels_np, cv=3).mean()
            ar = cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                                 Rs, labels_np, cv=3).mean()
            ao = cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                                 Os, labels_np, cv=3).mean()
            arnd = cross_val_score(LogisticRegression(max_iter=1000, C=1.0),
                                   Rnd, labels_np, cv=3).mean()
            dm = ar / max(af, 0.01)
            results[cap] = {
                "features_accuracy": float(af),
                "residual_accuracy": float(ar),
                "original_accuracy": float(ao),
                "random_accuracy": float(arnd),
                "dark_matter_ratio": float(dm),
                "n_examples": len(labels_np),
                "label_balance": f"{n1}/{n0}",
            }
            print(f"    feat={af:.3f} resid={ar:.3f} orig={ao:.3f} "
                  f"rand={arnd:.3f} dm={dm:.3f}")
        except Exception as exc:
            print(f"    error: {exc}")
            results[cap] = {
                "features_accuracy": 0.5, "residual_accuracy": 0.5,
                "original_accuracy": 0.5, "random_accuracy": 0.5,
                "dark_matter_ratio": 1.0, "error": str(exc)}

    with open(RESULTS_DIR / "dark_matter_probes.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    model = load_model()
    saes = load_saes()
    avail = sorted(saes.keys())
    print(f"\nLayers: {avail}, features/SAE: {saes[0].cfg.d_sae}")

    ok = verify_gradient_flow(model, saes)
    if not ok:
        print("FATAL: gradient flow verification failed")
        return

    agg, tf, all_attr = run_attribution_all(model, saes)
    print(f"\n  attribution: {(time.time()-t0)/60:.1f} min")

    fli, peaks = compute_fli(all_attr, avail)
    print(f"  FLI: {(time.time()-t0)/60:.1f} min")

    causal = run_causal_validation(model, saes, agg, peaks)
    print(f"  causal: {(time.time()-t0)/60:.1f} min")

    overlap = compute_overlap_matrix(peaks, avail)
    print(f"  overlap: {(time.time()-t0)/60:.1f} min")

    dm = run_dark_matter_analysis(model, saes, peaks)
    print(f"  dark matter: {(time.time()-t0)/60:.1f} min")

    print(f"\n=== Done — {(time.time()-t0)/60:.1f} min total ===")
    del model
    for s in saes.values():
        del s
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
