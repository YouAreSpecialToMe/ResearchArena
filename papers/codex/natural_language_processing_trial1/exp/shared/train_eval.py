import math
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .utils import (
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_GRAD_NORM,
    MAX_SEQ_LEN,
    MODEL_NAME,
    NUM_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    TARGET_MODULES,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    dump_json,
    save_run_config,
    set_seed,
)


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def example_token_length(example, tokenizer):
    return len(tokenizer(example["input_text"] + "\n" + example["target_text"], add_special_tokens=False)["input_ids"])


def pack_to_budget(items, tokenizer, budget_tokens):
    selected = []
    total = 0
    for item in items:
        tokens = example_token_length(item, tokenizer)
        if total + tokens <= budget_tokens or not selected:
            selected.append(item)
            total += tokens
    return selected, total


def interleave_lists(a_items, b_items):
    out = []
    max_len = max(len(a_items), len(b_items))
    for idx in range(max_len):
        if idx < len(a_items):
            out.append(a_items[idx])
        if idx < len(b_items):
            out.append(b_items[idx])
    return out


def compute_train_selection(data, tokenizer):
    logic_train = data["logic"]["train"]
    planning_train = data["planning"]["train"]
    logic_full = logic_train["full"][:96]
    plan_full = planning_train["full"][:96]
    logic_ud_clean = logic_train["clean"][:96]
    logic_ud_noisy = logic_train["noisy"][:96]
    plan_ud_clean = planning_train["clean"][:96]
    plan_ud_noisy = planning_train["noisy"][:96]

    clean_logic_budget = min(
        sum(example_token_length(item, tokenizer) for item in logic_ud_clean),
        sum(example_token_length(item, tokenizer) for item in logic_ud_noisy),
    )
    clean_plan_budget = min(
        sum(example_token_length(item, tokenizer) for item in plan_ud_clean),
        sum(example_token_length(item, tokenizer) for item in plan_ud_noisy),
    )

    logic_ud_clean, logic_ud_clean_tokens = pack_to_budget(logic_ud_clean, tokenizer, clean_logic_budget)
    logic_ud_noisy, logic_ud_noisy_tokens = pack_to_budget(logic_ud_noisy, tokenizer, clean_logic_budget)
    plan_ud_clean, plan_ud_clean_tokens = pack_to_budget(plan_ud_clean, tokenizer, clean_plan_budget)
    plan_ud_noisy, plan_ud_noisy_tokens = pack_to_budget(plan_ud_noisy, tokenizer, clean_plan_budget)

    logic_full_tokens = sum(example_token_length(item, tokenizer) for item in logic_full)
    plan_full_tokens = sum(example_token_length(item, tokenizer) for item in plan_full)
    logic_target_total = logic_full_tokens + min(logic_ud_clean_tokens, logic_ud_noisy_tokens)
    plan_target_total = plan_full_tokens + min(plan_ud_clean_tokens, plan_ud_noisy_tokens)
    answer_logic, answer_logic_tokens = pack_to_budget(logic_train["full"], tokenizer, logic_target_total)
    answer_plan, answer_plan_tokens = pack_to_budget(planning_train["full"], tokenizer, plan_target_total)

    answer_raw = answer_logic + answer_plan
    clean_logic_raw = logic_full + logic_ud_clean
    clean_plan_raw = plan_full + plan_ud_clean
    noisy_logic_raw = logic_full + logic_ud_noisy
    noisy_plan_raw = plan_full + plan_ud_noisy
    clean_raw_total = sum(example_token_length(item, tokenizer) for item in clean_logic_raw + clean_plan_raw)
    noisy_raw_total = sum(example_token_length(item, tokenizer) for item in noisy_logic_raw + noisy_plan_raw)
    answer_raw_total = answer_logic_tokens + answer_plan_tokens
    final_total = min(answer_raw_total, clean_raw_total, noisy_raw_total)
    avg_logic_total = (
        sum(example_token_length(item, tokenizer) for item in clean_logic_raw)
        + sum(example_token_length(item, tokenizer) for item in noisy_logic_raw)
    ) / 2
    logic_share = avg_logic_total / max(1.0, (clean_raw_total + noisy_raw_total) / 2)
    final_logic_budget = int(round(final_total * logic_share))
    final_plan_budget = final_total - final_logic_budget
    clean_logic, clean_logic_tokens = pack_to_budget(interleave_lists(logic_full, logic_ud_clean), tokenizer, final_logic_budget)
    clean_plan, clean_plan_tokens = pack_to_budget(interleave_lists(plan_full, plan_ud_clean), tokenizer, final_plan_budget)
    noisy_logic, noisy_logic_tokens = pack_to_budget(interleave_lists(logic_full, logic_ud_noisy), tokenizer, final_logic_budget)
    noisy_plan, noisy_plan_tokens = pack_to_budget(interleave_lists(plan_full, plan_ud_noisy), tokenizer, final_plan_budget)

    answer_selected, answer_selected_tokens = pack_to_budget(interleave_lists(answer_logic, answer_plan), tokenizer, final_total)
    realized_total = min(answer_selected_tokens, clean_logic_tokens + clean_plan_tokens, noisy_logic_tokens + noisy_plan_tokens)
    answer_selected, answer_selected_tokens = pack_to_budget(interleave_lists(answer_logic, answer_plan), tokenizer, realized_total)
    clean_logic, clean_logic_tokens = pack_to_budget(interleave_lists(logic_full, logic_ud_clean), tokenizer, int(round(realized_total * logic_share)))
    clean_plan, clean_plan_tokens = pack_to_budget(interleave_lists(plan_full, plan_ud_clean), tokenizer, realized_total - clean_logic_tokens)
    noisy_logic, noisy_logic_tokens = pack_to_budget(interleave_lists(logic_full, logic_ud_noisy), tokenizer, int(round(realized_total * logic_share)))
    noisy_plan, noisy_plan_tokens = pack_to_budget(interleave_lists(plan_full, plan_ud_noisy), tokenizer, realized_total - noisy_logic_tokens)
    final_target = min(answer_selected_tokens, clean_logic_tokens + clean_plan_tokens, noisy_logic_tokens + noisy_plan_tokens)
    answer_selected, answer_selected_tokens = pack_to_budget(interleave_lists(answer_logic, answer_plan), tokenizer, final_target)
    clean_logic, clean_logic_tokens = pack_to_budget(interleave_lists(logic_full, logic_ud_clean), tokenizer, int(round(final_target * logic_share)))
    clean_plan, clean_plan_tokens = pack_to_budget(interleave_lists(plan_full, plan_ud_clean), tokenizer, final_target - clean_logic_tokens)
    noisy_logic, noisy_logic_tokens = pack_to_budget(interleave_lists(logic_full, logic_ud_noisy), tokenizer, int(round(final_target * logic_share)))
    noisy_plan, noisy_plan_tokens = pack_to_budget(interleave_lists(plan_full, plan_ud_noisy), tokenizer, final_target - noisy_logic_tokens)
    selections = {
        "answer_only": answer_selected,
        "noisy": noisy_logic + noisy_plan,
        "clean": clean_logic + clean_plan,
    }
    stats = {
        "logic_ud_budget_tokens": min(logic_ud_clean_tokens, logic_ud_noisy_tokens),
        "planning_ud_budget_tokens": min(plan_ud_clean_tokens, plan_ud_noisy_tokens),
        "answer_only": {
            "logic_examples": sum(1 for item in answer_selected if item["domain"] == "logic"),
            "planning_examples": sum(1 for item in answer_selected if item["domain"] == "planning"),
            "token_total": answer_selected_tokens,
        },
        "noisy": {
            "logic_examples": len(noisy_logic),
            "planning_examples": len(noisy_plan),
            "token_total": noisy_logic_tokens + noisy_plan_tokens,
        },
        "clean": {
            "logic_examples": len(clean_logic),
            "planning_examples": len(clean_plan),
            "token_total": clean_logic_tokens + clean_plan_tokens,
        },
    }
    return selections, stats


def select_train_examples(data, condition, tokenizer=None):
    tokenizer = tokenizer or build_tokenizer()
    selections, _ = compute_train_selection(data, tokenizer)
    return selections[condition]


def select_dev_examples(data):
    return data["logic"]["validation"]["clean"][:40] + data["planning"]["validation"]["clean"][:40]


def tokenize_example(example, tokenizer):
    prompt_ids = tokenizer(example["input_text"], add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example["target_text"], add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id]
    input_ids = (prompt_ids + target_ids + eos)[:MAX_SEQ_LEN]
    label_start = min(len(prompt_ids), len(input_ids))
    labels = [-100] * label_start + input_ids[label_start:]
    labels = labels[: len(input_ids)]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_hf_dataset(examples, tokenizer):
    ds = Dataset.from_list(
        [{"input_text": ex["input_text"], "target_text": ex["target_text"]} for ex in examples]
    )
    return ds.map(lambda ex: tokenize_example(ex, tokenizer), remove_columns=["input_text", "target_text"])


class SupervisedCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = min(MAX_SEQ_LEN, max(len(feature["input_ids"]) for feature in features))
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_quantized_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def run_train(output_dir, data, condition, seed, example_limit=None):
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    model = build_quantized_model()
    train_examples = select_train_examples(data, condition, tokenizer=tokenizer)
    if example_limit is not None:
        train_examples = train_examples[:example_limit]
    dev_examples = select_dev_examples(data)
    train_ds = make_hf_dataset(train_examples, tokenizer)
    dev_ds = make_hf_dataset(dev_examples, tokenizer)
    collator = SupervisedCollator(tokenizer)
    steps_per_epoch = math.ceil(len(train_examples) / (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))
    config = save_run_config(
        output_dir / "config.json",
        condition=condition,
        seed=seed,
        extra={
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "example_limit": example_limit,
        },
    )
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoint"),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        logging_steps=max(1, steps_per_epoch // 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=seed,
        data_seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    t0 = time.time()
    train_result = trainer.train()
    runtime = time.time() - t0
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    peak_vram = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
    metrics = {
        "condition": condition,
        "seed": seed,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "runtime_seconds": runtime,
        "train_loss": float(train_result.training_loss),
        "peak_vram_gb": peak_vram,
        "examples_per_second": len(train_examples) / max(runtime, 1e-6),
        "tokens_per_second": sum(example_token_length(item, tokenizer) for item in train_examples) / max(runtime, 1e-6),
        "adapter_dir": str(adapter_dir),
        "config": config,
    }
    dump_json(metrics, output_dir / "train_metrics.json")
    return metrics


def load_trained_model(adapter_dir):
    tokenizer = build_tokenizer()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.config.use_cache = True
    return tokenizer, model


def strip_prompt(prompt, generated_text):
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    return generated_text.strip()


def batch_generate(adapter_dir, prompts, batch_size=8, max_new_tokens=24):
    tokenizer, model = load_trained_model(adapter_dir)
    return batch_generate_loaded(tokenizer, model, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens)


def batch_generate_loaded(tokenizer, model, prompts, batch_size=8, max_new_tokens=24):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        toks = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        outputs.extend([strip_prompt(prompt, text) for prompt, text in zip(batch, texts)])
    return outputs


def normalize_prediction(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_ask(text):
    text = text.strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return re.sub(r"\s+", " ", text.strip().lower())


def compute_full_accuracy(preds, examples):
    indicators = [
        int(normalize_prediction(pred) == normalize_prediction(ex["target_text"]))
        for pred, ex in zip(preds, examples)
    ]
    return sum(indicators) / max(1, len(indicators)), indicators


def is_ask(text):
    return text.strip().lower().startswith("ask:")


def compute_underspecified_metrics(preds, examples, tokenizer=None, model=None):
    ask_indicators = []
    question_indicators = []
    solve_after_indicators = [0 for _ in examples]
    clarification_prompts = []
    clarification_targets = []
    clarification_indices = []
    for idx, (pred, ex) in enumerate(zip(preds, examples)):
        wants_ask = ex.get("answer_required", ex["target_text"].lower().startswith("ask:"))
        pred_ask = is_ask(pred)
        ask_indicators.append(int(pred_ask == wants_ask))
        default_valid = []
        if ex.get("valid_asks"):
            default_valid = ex["valid_asks"]
        elif ex.get("gt_qs"):
            default_valid = ex["gt_qs"]
        elif ":" in ex.get("target_text", ""):
            default_valid = [ex["target_text"].split(":", 1)[1].strip()]
        valid_asks = [normalize_ask(item) for item in default_valid]
        if pred_ask:
            asked = normalize_ask(pred)
            question_ok = asked in valid_asks
        else:
            answer_target = normalize_prediction(ex.get("no_question_answer", "ANSWER: no_question_needed"))
            question_ok = (not wants_ask) and normalize_prediction(pred) == answer_target
        question_indicators.append(int(question_ok))
        if wants_ask and question_ok and ex.get("clarified_input_text") and ex.get("full_target_text"):
            clarification_indices.append(idx)
            clarification_prompts.append(ex["clarified_input_text"])
            clarification_targets.append(ex["full_target_text"])

    if clarification_prompts and tokenizer is not None and model is not None:
        clarified_preds = batch_generate_loaded(tokenizer, model, clarification_prompts, max_new_tokens=16)
        for idx, pred, target in zip(clarification_indices, clarified_preds, clarification_targets):
            solve_after_indicators[idx] = int(normalize_prediction(pred) == normalize_prediction(target))

    n = max(1, len(examples))
    return {
        "ask_answer_accuracy": sum(ask_indicators) / n,
        "question_accuracy": sum(question_indicators) / n,
        "solve_after_clarification_rate": sum(solve_after_indicators) / n,
        "per_example": {
            "ask_answer_accuracy": ask_indicators,
            "question_accuracy": question_indicators,
            "solve_after_clarification_rate": solve_after_indicators,
        },
    }


def compute_questbench_metrics(preds, examples, tokenizer=None, model=None):
    return compute_underspecified_metrics(preds, examples, tokenizer=tokenizer, model=model)
