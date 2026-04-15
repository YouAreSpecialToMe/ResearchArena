"""
Step 3: Ablation experiments (CORRECTED)
  - Architecture comparison: TopK-32K (primary) vs Standard-24K (JB release)
  - Attribution method comparison: grad×act vs activation-only vs gradient-only
  - Top-K sensitivity analysis

Uses capability-specific losses from step2 for syntax and semantic.
"""
import json
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

import transformer_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Import corrected functions from step2
from step2_main_experiments import (
    find_target_word_position, find_diverging_position,
    get_sentiment_token_ids, POS_WORDS, NEG_WORDS
)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAPABILITIES = ["factual", "syntax", "sentiment", "semantic", "ner", "reasoning"]
SEEDS = [42, 123, 456]
TOP_K = 100
N_ATTR = 300
BATCH_SIZE = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_capability_data(cap):
    with open(DATA_DIR / f"{cap}.json") as f:
        return json.load(f)


def get_text(cap, ex):
    if cap == "syntax":
        return ex["good_sentence"]
    elif cap == "semantic":
        return ex["sentence1"]
    return ex["prompt"]


def cap_loss_standard(cap, model, logits, tokens, examples, pos_ids=None, neg_ids=None):
    """Capability loss for factual/NER/reasoning/sentiment (single forward pass)."""
    if cap in ("factual", "ner", "reasoning"):
        last = logits[:, -1, :]
        tgt = torch.tensor([e["target_token_id"] for e in examples], device=device)
        lp = F.log_softmax(last, dim=-1)
        return -lp[torch.arange(len(tgt)), tgt].mean()
    if cap == "sentiment":
        last = logits[:, -1, :]
        lp = F.log_softmax(last, dim=-1)
        if pos_ids and neg_ids:
            pos_ids_t = torch.tensor(pos_ids, device=device)
            neg_ids_t = torch.tensor(neg_ids, device=device)
            losses = []
            for i, e in enumerate(examples):
                if e["label"] == 1:
                    losses.append(-torch.logsumexp(lp[i, pos_ids_t], dim=0))
                else:
                    losses.append(-torch.logsumexp(lp[i, neg_ids_t], dim=0))
            return torch.stack(losses).mean()
        else:
            pos = model.to_single_token(" good")
            neg = model.to_single_token(" bad")
            losses = []
            for i, e in enumerate(examples):
                losses.append(-lp[i, pos if e["label"] == 1 else neg])
            return torch.stack(losses).mean()
    raise ValueError(f"cap_loss_standard doesn't handle {cap}")


# ── Architecture comparison ──────────────────────────────────────────────────

def architecture_comparison(model):
    """
    Compare FLI from primary TopK-32K SAEs (resid_post) with Standard-24K
    SAEs from the JB release (resid_pre).
    """
    print("\n=== Architecture Comparison: TopK-32K vs Standard-24K ===")

    with open(RESULTS_DIR / "peak_layers.json") as f:
        peak_layers = json.load(f)

    pos_ids, neg_ids = get_sentiment_token_ids(model)

    layers_needed = set()
    for cap in CAPABILITIES:
        pl = peak_layers[cap]
        for dl in (-1, 0, 1):
            l = pl + dl
            if 0 <= l <= 10:
                layers_needed.add(l)
    layers_needed = sorted(layers_needed)

    primary_saes = {}
    alt_saes = {}
    alt_hook_layers = {}

    for l in layers_needed:
        sae_p = SAE.from_pretrained(
            release="gpt2-small-resid-post-v5-32k",
            sae_id=f"blocks.{l}.hook_resid_post", device=str(device))
        if isinstance(sae_p, tuple):
            sae_p = sae_p[0]
        primary_saes[l] = sae_p

        jb_layer = l + 1
        try:
            sae_a = SAE.from_pretrained(
                release="gpt2-small-res-jb",
                sae_id=f"blocks.{jb_layer}.hook_resid_pre",
                device=str(device))
            if isinstance(sae_a, tuple):
                sae_a = sae_a[0]
            alt_saes[l] = sae_a
            alt_hook_layers[l] = jb_layer
            if l == layers_needed[0]:
                print(f"  Primary: d_sae={sae_p.cfg.d_sae}")
                print(f"  Alt:     d_sae={sae_a.cfg.d_sae}")
        except Exception as e:
            print(f"  JB layer {jb_layer}: {e}")

    if not alt_saes:
        print("  No alternative SAEs loaded — skipping.")
        return {}

    fli_primary = {}
    fli_alt = {}

    set_seed(42)
    for cap in CAPABILITIES:
        data = load_capability_data(cap)
        idx = random.sample(range(len(data)), min(N_ATTR, len(data)))
        subset = [data[i] for i in idx]
        pl = peak_layers[cap]

        for dl in (-1, 0, 1):
            l = pl + dl
            if l not in primary_saes or l not in alt_saes:
                continue

            hp_p = f"blocks.{l}.hook_resid_post"
            hp_a = f"blocks.{alt_hook_layers[l]}.hook_resid_pre"
            sae_p = primary_saes[l]
            sae_a = alt_saes[l]

            imp_p = torch.zeros(sae_p.cfg.d_sae, device=device)
            imp_a = torch.zeros(sae_a.cfg.d_sae, device=device)
            ok = 0

            for ex in tqdm(subset, desc=f"  {cap} L{l}", leave=False):
                try:
                    if cap in ("factual", "ner", "reasoning", "sentiment"):
                        text = get_text(cap, ex)
                        tokens = model.to_tokens(text, prepend_bos=True)
                        if tokens.shape[1] > 128:
                            tokens = tokens[:, :128]

                        stored = {}
                        def _mk(name):
                            def fn(v, hook=None):
                                stored[name] = v
                                return v
                            return fn

                        logits = model.run_with_hooks(tokens, fwd_hooks=[
                            (hp_p, _mk("p")), (hp_a, _mk("a"))])
                        loss = cap_loss_standard(cap, model, logits, tokens, [ex],
                                                  pos_ids, neg_ids)

                        gp, ga = torch.autograd.grad(
                            loss, [stored["p"], stored["a"]], allow_unused=True)

                        if gp is not None:
                            gl = gp[:, -1, :].detach()
                            rl = stored["p"][:, -1, :].detach()
                            with torch.no_grad():
                                acts = sae_p.encode(rl)
                            proj = torch.einsum("bd,fd->bf", gl, sae_p.W_dec.detach())
                            imp_p += (acts * proj).abs().squeeze(0)

                        if ga is not None:
                            gl = ga[:, -1, :].detach()
                            rl = stored["a"][:, -1, :].detach()
                            with torch.no_grad():
                                acts = sae_a.encode(rl)
                            proj = torch.einsum("bd,fd->bf", gl, sae_a.W_dec.detach())
                            imp_a += (acts * proj).abs().squeeze(0)

                    elif cap == "syntax":
                        # BLiMP: loss at critical diverging position
                        good_tok = model.to_tokens(ex["good_sentence"], prepend_bos=True)
                        bad_tok = model.to_tokens(ex["bad_sentence"], prepend_bos=True)
                        if good_tok.shape[1] > 128: good_tok = good_tok[:, :128]
                        if bad_tok.shape[1] > 128: bad_tok = bad_tok[:, :128]

                        div_pos = find_diverging_position(good_tok, bad_tok)
                        pred_pos = max(div_pos - 1, 0)

                        stored_g = {}
                        def _mkg(name):
                            def fn(v, hook=None):
                                stored_g[name] = v
                                return v
                            return fn

                        lg = model.run_with_hooks(good_tok, fwd_hooks=[
                            (hp_p, _mkg("p")), (hp_a, _mkg("a"))])

                        lp = F.log_softmax(lg[0, pred_pos, :], dim=-1)
                        good_tok_id = good_tok[0, div_pos]
                        bad_tok_id = bad_tok[0, div_pos] if div_pos < bad_tok.shape[1] else good_tok_id
                        loss = -(lp[good_tok_id] - lp[bad_tok_id])

                        all_tensors = [stored_g.get("p"), stored_g.get("a")]
                        valid = [t for t in all_tensors if t is not None]
                        grads = torch.autograd.grad(loss, valid, allow_unused=True)
                        grad_idx = 0
                        for name, sae_obj, imp_ref in [("p", sae_p, "p"), ("a", sae_a, "a")]:
                            if name in stored_g:
                                g = grads[grad_idx]; grad_idx += 1
                                if g is not None:
                                    gl_ = g[:, pred_pos, :].detach()
                                    rl_ = stored_g[name][:, pred_pos, :].detach()
                                    with torch.no_grad():
                                        acts_ = sae_obj.encode(rl_)
                                    proj_ = torch.einsum("bd,fd->bf", gl_, sae_obj.W_dec.detach())
                                    if imp_ref == "p":
                                        imp_p += (acts_ * proj_).abs().squeeze(0)
                                    else:
                                        imp_a += (acts_ * proj_).abs().squeeze(0)

                    elif cap == "semantic":
                        # WiC cosine similarity loss
                        tok1 = model.to_tokens(ex["sentence1"], prepend_bos=True)
                        tok2 = model.to_tokens(ex["sentence2"], prepend_bos=True)
                        if tok1.shape[1] > 128: tok1 = tok1[:, :128]
                        if tok2.shape[1] > 128: tok2 = tok2[:, :128]

                        stored1, stored2 = {}, {}
                        def _mk1(name):
                            def fn(v, hook=None):
                                stored1[name] = v
                                return v
                            return fn
                        def _mk2(name):
                            def fn(v, hook=None):
                                stored2[name] = v
                                return v
                            return fn

                        model.run_with_hooks(tok1, fwd_hooks=[
                            (hp_p, _mk1("p")), (hp_a, _mk1("a"))])
                        model.run_with_hooks(tok2, fwd_hooks=[
                            (hp_p, _mk2("p")), (hp_a, _mk2("a"))])

                        # Use primary hook point for loss
                        pos1 = find_target_word_position(model.tokenizer, ex["sentence1"], ex["target_word"])
                        pos2 = find_target_word_position(model.tokenizer, ex["sentence2"], ex["target_word"])
                        pos1 = min(pos1 + 1, tok1.shape[1] - 1)
                        pos2 = min(pos2 + 1, tok2.shape[1] - 1)

                        r1 = stored1["p"][:, pos1, :]
                        r2 = stored2["p"][:, pos2, :]
                        cs = F.cosine_similarity(r1, r2, dim=-1)
                        loss = cs.mean() if not ex["same_sense"] else (1.0 - cs.mean())

                        all_tensors = [stored1.get("p"), stored1.get("a"),
                                       stored2.get("p"), stored2.get("a")]
                        valid = [t for t in all_tensors if t is not None]
                        grads = torch.autograd.grad(loss, valid, allow_unused=True)
                        grad_idx = 0
                        for name, stored_dict, sae_obj, imp_ref, pos in [
                            ("p", stored1, sae_p, "p", pos1), ("a", stored1, sae_a, "a", pos1),
                            ("p", stored2, sae_p, "p", pos2), ("a", stored2, sae_a, "a", pos2)]:
                            if name in stored_dict:
                                g = grads[grad_idx]; grad_idx += 1
                                if g is not None:
                                    gl_ = g[:, pos, :].detach()
                                    rl_ = stored_dict[name][:, pos, :].detach()
                                    with torch.no_grad():
                                        acts_ = sae_obj.encode(rl_)
                                    proj_ = torch.einsum("bd,fd->bf", gl_, sae_obj.W_dec.detach())
                                    if imp_ref == "p":
                                        imp_p += (acts_ * proj_).abs().squeeze(0)
                                    else:
                                        imp_a += (acts_ * proj_).abs().squeeze(0)

                    ok += 1
                except Exception:
                    continue
                finally:
                    torch.cuda.empty_cache()

            if ok > 0:
                imp_p /= ok
                imp_a /= ok

            def _fli(imp):
                p = imp.float() + 1e-10
                p = p / p.sum()
                H = -(p * p.log()).sum().item()
                return 1.0 - H / np.log(len(p))

            key = f"{cap}_L{l}"
            fli_primary[key] = _fli(imp_p)
            fli_alt[key] = _fli(imp_a)

    # Spearman correlation
    common = sorted(set(fli_primary) & set(fli_alt))
    if len(common) >= 4:
        x = [fli_primary[k] for k in common]
        y = [fli_alt[k] for k in common]
        rho, pval = scipy_stats.spearmanr(x, y)
    else:
        rho, pval = 0.0, 1.0

    res = {
        "primary_type": "TopK-32K (resid_post)",
        "alt_type": "Standard-24K (JB, resid_pre shifted)",
        "fli_primary": fli_primary,
        "fli_alt": fli_alt,
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n_comparisons": len(common),
    }
    print(f"\n  Spearman ρ = {rho:.3f} (p = {pval:.4f}, n = {len(common)})")

    with open(RESULTS_DIR / "architecture_comparison.json", "w") as f:
        json.dump(res, f, indent=2)

    for s in list(primary_saes.values()) + list(alt_saes.values()):
        del s
    torch.cuda.empty_cache()
    gc.collect()
    return res


# ── Attribution method comparison (CORRECTED) ───────────────────────────────

def attribution_method_comparison(model):
    """
    Compare grad×act vs activation-only vs gradient-only vs random.
    Uses capability-specific losses.
    """
    print("\n=== Attribution Method Comparison ===")

    with open(RESULTS_DIR / "peak_layers.json") as f:
        peak_layers = json.load(f)

    pos_ids, neg_ids = get_sentiment_token_ids(model)

    pk_set = set(peak_layers.values())
    saes = {}
    for l in pk_set:
        sae = SAE.from_pretrained(
            release="gpt2-small-resid-post-v5-32k",
            sae_id=f"blocks.{l}.hook_resid_post", device=str(device))
        if isinstance(sae, tuple):
            sae = sae[0]
        saes[l] = sae

    methods = ["grad_x_act", "activation_only", "gradient_only", "random"]
    results = {}

    for cap in CAPABILITIES:
        print(f"\n  {cap}")
        pl = peak_layers[cap]
        sae = saes[pl]
        hp = f"blocks.{pl}.hook_resid_post"

        data = load_capability_data(cap)
        set_seed(42)
        idx = random.sample(range(len(data)), min(N_ATTR, len(data)))
        subset = [data[i] for i in idx]

        d_sae = sae.cfg.d_sae
        imp = {m: torch.zeros(d_sae, device=device) for m in methods}
        ok = 0

        for ex in tqdm(subset, desc=f"    attr", leave=False):
            try:
                if cap in ("factual", "ner", "reasoning", "sentiment"):
                    text = get_text(cap, ex)
                    tokens = model.to_tokens(text, prepend_bos=True)
                    if tokens.shape[1] > 128:
                        tokens = tokens[:, :128]

                    stored = {}
                    def _hook(v, hook=None):
                        stored["r"] = v
                        return v

                    logits = model.run_with_hooks(tokens, fwd_hooks=[(hp, _hook)])
                    loss = cap_loss_standard(cap, model, logits, tokens, [ex],
                                              pos_ids, neg_ids)
                    (g,) = torch.autograd.grad(loss, [stored["r"]])

                    gl = g[:, -1, :].detach()
                    rl = stored["r"][:, -1, :].detach()
                    with torch.no_grad():
                        acts = sae.encode(rl).squeeze(0)
                    W = sae.W_dec.detach()
                    proj = torch.einsum("d,fd->f", gl.squeeze(0), W)

                elif cap == "syntax":
                    good_tok = model.to_tokens(ex["good_sentence"], prepend_bos=True)
                    bad_tok = model.to_tokens(ex["bad_sentence"], prepend_bos=True)
                    if good_tok.shape[1] > 128: good_tok = good_tok[:, :128]
                    if bad_tok.shape[1] > 128: bad_tok = bad_tok[:, :128]

                    div_pos = find_diverging_position(good_tok, bad_tok)
                    pred_pos = max(div_pos - 1, 0)

                    stored_g = {}
                    def _hg(v, hook=None):
                        stored_g["r"] = v; return v

                    lg = model.run_with_hooks(good_tok, fwd_hooks=[(hp, _hg)])

                    lp_s = F.log_softmax(lg[0, pred_pos, :], dim=-1)
                    good_tok_id = good_tok[0, div_pos]
                    bad_tok_id = bad_tok[0, div_pos] if div_pos < bad_tok.shape[1] else good_tok_id
                    loss = -(lp_s[good_tok_id] - lp_s[bad_tok_id])

                    (gg,) = torch.autograd.grad(loss, [stored_g["r"]])

                    gl = gg[:, pred_pos, :].detach()
                    rl = stored_g["r"][:, pred_pos, :].detach()
                    with torch.no_grad():
                        acts = sae.encode(rl).squeeze(0)
                    W = sae.W_dec.detach()
                    proj = torch.einsum("d,fd->f", gl.squeeze(0), W)

                elif cap == "semantic":
                    tok1 = model.to_tokens(ex["sentence1"], prepend_bos=True)
                    tok2 = model.to_tokens(ex["sentence2"], prepend_bos=True)
                    if tok1.shape[1] > 128: tok1 = tok1[:, :128]
                    if tok2.shape[1] > 128: tok2 = tok2[:, :128]

                    stored1, stored2 = {}, {}
                    def _h1(v, hook=None):
                        stored1["r"] = v; return v
                    def _h2(v, hook=None):
                        stored2["r"] = v; return v

                    model.run_with_hooks(tok1, fwd_hooks=[(hp, _h1)])
                    model.run_with_hooks(tok2, fwd_hooks=[(hp, _h2)])

                    pos1 = find_target_word_position(model.tokenizer, ex["sentence1"], ex["target_word"])
                    pos2 = find_target_word_position(model.tokenizer, ex["sentence2"], ex["target_word"])
                    pos1 = min(pos1 + 1, tok1.shape[1] - 1)
                    pos2 = min(pos2 + 1, tok2.shape[1] - 1)

                    r1 = stored1["r"][:, pos1, :]
                    r2 = stored2["r"][:, pos2, :]
                    cs = F.cosine_similarity(r1, r2, dim=-1)
                    loss = cs.mean() if not ex["same_sense"] else (1.0 - cs.mean())

                    g1, g2 = torch.autograd.grad(loss, [stored1["r"], stored2["r"]], allow_unused=True)

                    gl = g1[:, pos1, :].detach() if g1 is not None else torch.zeros(1, sae.cfg.d_in, device=device)
                    rl = stored1["r"][:, pos1, :].detach()
                    with torch.no_grad():
                        acts = sae.encode(rl).squeeze(0)
                    W = sae.W_dec.detach()
                    proj = torch.einsum("d,fd->f", gl.squeeze(0), W)

                imp["grad_x_act"] += (acts * proj).abs()
                imp["activation_only"] += acts.abs()
                imp["gradient_only"] += proj.abs()
                imp["random"] += torch.randn(d_sae, device=device).abs()
                ok += 1
            except Exception:
                continue
            finally:
                torch.cuda.empty_cache()

        if ok > 0:
            for m in methods:
                imp[m] /= ok

        # Causal fidelity with capability-specific scoring
        set_seed(42)
        eval_idx = random.sample(range(len(data)), min(200, len(data)))
        eval_sub = [data[i] for i in eval_idx]

        def _score(ablate=None):
            """Score function respecting capability-specific metrics."""
            total, n = 0.0, 0
            for ii in range(0, len(eval_sub), BATCH_SIZE):
                be = eval_sub[ii:ii + BATCH_SIZE]

                if cap in ("factual", "ner", "reasoning"):
                    bt = [e["prompt"] for e in be]
                    toks = model.to_tokens(bt, prepend_bos=True)
                    if toks.shape[1] > 128: toks = toks[:, :128]
                    if ablate is not None:
                        def _h(resid, hook=None):
                            b, s, d = resid.shape
                            flat = resid.view(-1, d)
                            a = sae.encode(flat)
                            a[:, ablate] = 0
                            return sae.decode(a).view(b, s, d)
                        with torch.no_grad():
                            lo = model.run_with_hooks(toks, fwd_hooks=[(hp, _h)])
                    else:
                        with torch.no_grad():
                            lo = model(toks)
                    for j, e in enumerate(be):
                        p = F.softmax(lo[j, -1, :], dim=-1)
                        total += p[e["target_token_id"]].item()
                    n += len(be)

                elif cap == "sentiment":
                    bt = [e["prompt"] for e in be]
                    toks = model.to_tokens(bt, prepend_bos=True)
                    if toks.shape[1] > 128: toks = toks[:, :128]
                    if ablate is not None:
                        def _h(resid, hook=None):
                            b, s, d = resid.shape
                            flat = resid.view(-1, d)
                            a = sae.encode(flat)
                            a[:, ablate] = 0
                            return sae.decode(a).view(b, s, d)
                        with torch.no_grad():
                            lo = model.run_with_hooks(toks, fwd_hooks=[(hp, _h)])
                    else:
                        with torch.no_grad():
                            lo = model(toks)
                    for j, e in enumerate(be):
                        p = F.softmax(lo[j, -1, :], dim=-1)
                        if pos_ids and neg_ids:
                            pp = sum(p[t].item() for t in pos_ids)
                            np_ = sum(p[t].item() for t in neg_ids)
                            total += pp / (pp + np_ + 1e-10) if e["label"] == 1 else np_ / (pp + np_ + 1e-10)
                        else:
                            total += 0.5
                    n += len(be)

                elif cap == "syntax":
                    for e in be:
                        good_tok = model.to_tokens(e["good_sentence"], prepend_bos=True)
                        bad_tok = model.to_tokens(e["bad_sentence"], prepend_bos=True)
                        if good_tok.shape[1] > 128: good_tok = good_tok[:, :128]
                        if bad_tok.shape[1] > 128: bad_tok = bad_tok[:, :128]
                        dv = find_diverging_position(good_tok, bad_tok)
                        pp = max(dv - 1, 0)
                        if ablate is not None:
                            def _h(resid, hook=None):
                                b, s, d = resid.shape
                                flat = resid.view(-1, d)
                                a = sae.encode(flat)
                                a[:, ablate] = 0
                                return sae.decode(a).view(b, s, d)
                            with torch.no_grad():
                                lg_ = model.run_with_hooks(good_tok, fwd_hooks=[(hp, _h)])
                        else:
                            with torch.no_grad():
                                lg_ = model(good_tok)
                        lp_ = F.log_softmax(lg_[0, pp, :], dim=-1)
                        gtid = good_tok[0, dv].item()
                        btid = bad_tok[0, dv].item() if dv < bad_tok.shape[1] else gtid
                        total += (lp_[gtid].item() - lp_[btid].item())
                        n += 1

                elif cap == "semantic":
                    for e in be:
                        tok1 = model.to_tokens(e["sentence1"], prepend_bos=True)
                        tok2 = model.to_tokens(e["sentence2"], prepend_bos=True)
                        if tok1.shape[1] > 128: tok1 = tok1[:, :128]
                        if tok2.shape[1] > 128: tok2 = tok2[:, :128]
                        if ablate is not None:
                            def _h(resid, hook=None):
                                b, s, d = resid.shape
                                flat = resid.view(-1, d)
                                a_ = sae.encode(flat)
                                a_[:, ablate] = 0
                                return sae.decode(a_).view(b, s, d)
                            with torch.no_grad():
                                _, c1 = model.run_with_cache(tok1, names_filter=[hp])
                                _, c2 = model.run_with_cache(tok2, names_filter=[hp])
                            r1_ = c1[hp][:, -1, :]  # simplified: use last position
                            r2_ = c2[hp][:, -1, :]
                            a1_ = sae.encode(r1_); a1_[:, ablate] = 0
                            a2_ = sae.encode(r2_); a2_[:, ablate] = 0
                            sim = F.cosine_similarity(sae.decode(a1_), sae.decode(a2_), dim=-1).item()
                        else:
                            with torch.no_grad():
                                _, c1 = model.run_with_cache(tok1, names_filter=[hp])
                                _, c2 = model.run_with_cache(tok2, names_filter=[hp])
                            r1_ = c1[hp][:, -1, :]
                            r2_ = c2[hp][:, -1, :]
                            sim = F.cosine_similarity(r1_, r2_, dim=-1).item()
                        pred_same = sim > 0.99
                        total += (1 if pred_same == e["same_sense"] else 0)
                        n += 1

            return total / max(n, 1)

        base = _score()
        rand_score = _score(random.sample(range(d_sae), 50))
        cap_res = {"baseline": float(base)}
        for m in methods:
            top50 = imp[m].argsort(descending=True)[:50].tolist()
            s = _score(top50)
            drop = base - s
            d_rand = base - rand_score
            raw_fid = abs(drop) / (abs(d_rand) + 1e-6)
            cap_res[m] = {
                "score_top50_ablated": float(s),
                "drop": float(drop),
                "causal_fidelity": float(min(raw_fid, 100.0)),
            }
            print(f"    {m:>18s}: drop={drop:.4f} fid={min(raw_fid, 100.0):.2f}")
        results[cap] = cap_res

    with open(RESULTS_DIR / "attribution_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    for s in saes.values():
        del s
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ── Top-K sensitivity ────────────────────────────────────────────────────────

def topk_sensitivity():
    """Test sensitivity of FLI and overlap to the choice of K."""
    print("\n=== Top-K Sensitivity ===")

    agg = torch.load(RESULTS_DIR / "attribution_aggregated.pt",
                     weights_only=False)
    with open(RESULTS_DIR / "peak_layers.json") as f:
        peak_layers = json.load(f)

    Ks = [10, 25, 50, 100, 200, 500]
    fli_by_k = {str(k): {} for k in Ks}
    overlap_by_k = {str(k): {} for k in Ks}

    for cap in CAPABILITIES:
        pl = peak_layers[cap]
        if pl not in agg[cap]:
            continue
        mi = agg[cap][pl]["mean"]

        for K in Ks:
            topk_vals, _ = mi.topk(min(K, len(mi)))
            p = topk_vals + 1e-10
            p = p / p.sum()
            H = -(p * p.log()).sum().item()
            H_u = np.log(len(p))
            fli_by_k[str(K)][cap] = 1.0 - H / H_u

    # Rank stability
    rank_corr = {}
    ref = fli_by_k["100"]
    for K in Ks:
        cur = fli_by_k[str(K)]
        common = sorted(set(ref) & set(cur))
        if len(common) >= 4:
            r, p = scipy_stats.spearmanr(
                [ref[c] for c in common], [cur[c] for c in common])
            rank_corr[str(K)] = {"rho": float(r), "p": float(p)}
        else:
            rank_corr[str(K)] = {"rho": 0, "p": 1}

    # Overlap matrices
    with open(RESULTS_DIR / "top_features.json") as f:
        tf = json.load(f)
    for K in Ks:
        n = len(CAPABILITIES)
        ov = np.zeros((n, n))
        for i, ci in enumerate(CAPABILITIES):
            for j, cj in enumerate(CAPABILITIES):
                pi = str(peak_layers[ci])
                pj = str(peak_layers[cj])
                if pi in tf.get(ci, {}) and pj in tf.get(cj, {}):
                    si = set(tf[ci][pi][:K])
                    sj = set(tf[cj][pj][:K])
                    ov[i, j] = len(si & sj) / max(K, 1)
        overlap_by_k[str(K)] = ov.tolist()

    res = {
        "K_values": Ks,
        "fli_by_k": fli_by_k,
        "rank_correlation_vs_K100": rank_corr,
        "overlap_by_k": overlap_by_k,
    }
    with open(RESULTS_DIR / "topk_sensitivity.json", "w") as f:
        json.dump(res, f, indent=2)

    print("  Rank correlation (vs K=100):")
    for K in Ks:
        r = rank_corr[str(K)]
        print(f"    K={K:>4d}: ρ={r['rho']:.3f} (p={r['p']:.4f})")
    return res


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading model …")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    arch_res = architecture_comparison(model)
    print(f"\n  architecture: {(time.time()-t0)/60:.1f} min")

    attr_res = attribution_method_comparison(model)
    print(f"  attr methods: {(time.time()-t0)/60:.1f} min")

    topk_res = topk_sensitivity()
    print(f"  top-K: {(time.time()-t0)/60:.1f} min")

    print(f"\n=== Ablations done — {(time.time()-t0)/60:.1f} min ===")
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
