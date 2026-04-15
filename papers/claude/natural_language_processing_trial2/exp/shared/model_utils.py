"""Model loading and inference utilities."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np


def load_model(model_name, device="cuda"):
    """Load model and tokenizer in float16."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory."""
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def generate_text(model, tokenizer, prompts, max_new_tokens=256, temperature=0.7,
                  top_p=0.9, batch_size=4, do_sample=True):
    """Generate text for a list of prompts."""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Format as chat messages
        formatted = []
        for p in batch:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": p}]
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    text = p
            else:
                text = p
            formatted.append(text)

        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            results.append(generated.strip())

    return results


def get_yes_no_logprobs(model, tokenizer, prompts, batch_size=16):
    """Get P(Yes) for a list of yes/no prompts via logprobs.

    Uses the logits at the last non-padding position to extract P(Yes|prompt).
    With left-padding, we must find each sequence's true last token position.
    """
    confidences = []
    # Find Yes/No token IDs - include all common casing/spacing variants
    yes_ids = set()
    no_ids = set()
    for w in ["Yes", "yes", " Yes", " yes", "YES", "True", "true", " True"]:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for w in ["No", "no", " No", " no", "NO", "False", "false", " False"]:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])

    yes_ids = list(yes_ids)
    no_ids = list(no_ids)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = []
        for p in batch:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": p}]
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    text = p
            else:
                text = p
            formatted.append(text)

        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            all_logits = outputs.logits

        for j in range(len(batch)):
            # With left-padding, find the actual last token position correctly
            # attention_mask is 0 for pad tokens, 1 for real tokens
            # With left-padding, real tokens are right-aligned
            attn_mask = inputs["attention_mask"][j]
            # Last non-pad position = last index where attention_mask == 1
            # This is equivalent to total_length - 1 when padding is on the left
            nonpad_positions = attn_mask.nonzero(as_tuple=True)[0]
            if len(nonpad_positions) == 0:
                confidences.append(0.5)
                continue
            last_pos = nonpad_positions[-1].item()
            logit_vec = all_logits[j, last_pos, :]

            # Compute P(Yes) vs P(No) using logsumexp in log-space to avoid overflow.
            # The raw logits can be very large (>30), so exp() overflows to inf.
            # Instead, compute logsumexp over yes tokens and no tokens separately,
            # then use the log-space subtraction trick for the ratio.
            yes_logit_vals = logit_vec[yes_ids] if yes_ids else torch.tensor([-100.0]).to(logit_vec.device)
            no_logit_vals = logit_vec[no_ids] if no_ids else torch.tensor([-100.0]).to(logit_vec.device)

            log_yes = torch.logsumexp(yes_logit_vals, dim=0)
            log_no = torch.logsumexp(no_logit_vals, dim=0)

            # P(Yes) = exp(log_yes) / (exp(log_yes) + exp(log_no))
            #        = 1 / (1 + exp(log_no - log_yes))
            #        = sigmoid(log_yes - log_no)
            conf = torch.sigmoid(log_yes - log_no).item()

            if np.isnan(conf) or np.isinf(conf):
                conf = 0.5
            confidences.append(conf)

    return confidences


def get_token_logprobs(model, tokenizer, texts, batch_size=16):
    """Get average token log-probability for each text."""
    avg_logprobs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=256
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        for j in range(len(batch)):
            input_ids = inputs["input_ids"][j]
            mask = inputs["attention_mask"][j]
            log_probs = torch.log_softmax(logits[j], dim=-1)
            # Get log prob of each actual token
            token_logprobs = []
            for t in range(1, mask.sum().item()):
                token_id = input_ids[t]
                lp = log_probs[t-1, token_id].item()
                token_logprobs.append(lp)
            avg_lp = np.mean(token_logprobs) if token_logprobs else -10.0
            avg_logprobs.append(avg_lp)
    return avg_logprobs


def parse_claims(text):
    """Parse atomic claims from decomposition output."""
    claims = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Remove numbering like "1.", "- ", "* "
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        if len(line.split()) >= 4 and len(line) > 15:  # Minimum claim length
            claims.append(line)
    return claims
