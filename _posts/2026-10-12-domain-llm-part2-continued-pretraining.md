---
layout: post
title: "Teaching a Small LLM Your Domain — Part 2: Continued Pretraining"
description: "Code-first continued pretraining (CPT): build a text JSONL corpus, attach LoRA, and run causal language modeling with Hugging Face Trainer."
keywords: "continued pretraining CPT tutorial, LoRA Hugging Face Trainer, domain adaptation LLM, causal language modeling, PEFT, JSONL corpus, Qwen fine-tuning"
date: 2026-10-12
comments: true
series: domain-llm
part: 2
sitemap: true
tags:
  - LLM
  - LoRA
  - CPT
  - PyTorch
  - Hugging Face
  - series
glossary:
  - Continued Pretraining
  - LoRA
  - Causal Language Model
  - PEFT
  - Chat Template
  - Next-Token Prediction
---

[Continued pretraining (CPT)](/ai-glossary/continued-pretraining/) is ordinary [causal language modeling](/ai-glossary/causal-language-model/) on **your** text. No [chat template](/ai-glossary/chat-template/). No “assistant:” lines. Just [next-token prediction](/ai-glossary/next-token-prediction/) over domain documents so a small instruct model absorbs that domain before you teach it manners with [SFT](/ai-glossary/supervised-fine-tuning/).

This post is the code path: corpus → dataset → LoRA → `Trainer`.

## 1. Corpus format

One JSON object per line. The trainer only needs `text`; keep ids for debugging.

```json
{"text": "Title: Chapter I\nBook-id: wuthering_heights\n\n1801.—I have just returned from a visit to my landlord…", "book_id": "wuthering_heights", "chapter_id": "ch01"}
```

Build it from whatever chapter files you already have:

```python
import json
from pathlib import Path

def write_cpt_jsonl(
    chapters: list[dict],
    out_path: Path,
    max_chars: int = 12_000,
) -> int:
    """chapters: [{book_id, chapter_id, title, text}, ...]"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for ch in chapters:
            body = (ch.get("text") or "").strip()
            if len(body) < 80:
                continue
            body = body[:max_chars]
            header = (
                f"Title: {ch.get('title', '').strip()}\n"
                f"Chapter-id: {ch.get('chapter_id', '')}\n"
                f"Book-id: {ch.get('book_id', '')}\n\n"
            )
            row = {
                "text": header + body,
                "book_id": ch.get("book_id"),
                "chapter_id": ch.get("chapter_id"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
```

Why a short header? It gives the model light provenance inside the stream without turning CPT into instruction tuning.

Skip license pages and empty stubs. Prefer **chapter-sized** units over whole novels in one sample so truncation is predictable.

## 2. Load texts for CPT (not chat)

```python
import json
from pathlib import Path

def load_cpt_texts(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    if not texts:
        raise SystemExit(f"No texts in {path}")
    return texts
```

Contrast with SFT later: SFT rows use `messages` and `tokenizer.apply_chat_template(...)`. CPT must **not** wrap chapters in fake user/assistant turns.

## 3. Minimal causal-LM dataset

Labels equal input ids — standard LM objective. This avoids a `datasets` / pyarrow dependency when you want a thin script:

```python
class TextDataset:
    def __init__(self, texts: list[str], tokenizer, max_length: int = 1024):
        self.rows = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            ids = enc["input_ids"]
            self.rows.append(
                {
                    "input_ids": ids,
                    "attention_mask": enc.get("attention_mask")
                    or [1] * len(ids),
                    "labels": list(ids),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]
```

**Caveat:** tokenizing everything up front is simple and memory-hungry. Fine for tens of thousands of chapters on a training box; for millions of docs, stream or cache Arrow shards instead.

## 4. Attach LoRA

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_for_cpt(
    base: str,
    device: str,
    *,
    full_lora: bool = True,
    lora_r: int = 16,
):
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    targets = (
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if full_lora
        else ["q_proj", "v_proj"]  # laptop smoke
    )
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    if device != "cuda":
        model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer
```

Rule of thumb:

- **Smoke (MPS):** `full_lora=False`, `r=8`, `max_length=512`, few dozen steps.
- **Real CPT (CUDA):** `full_lora=True`, `r=16`, `max_length=1024+`, ~one epoch.

## 5. Train with Hugging Face `Trainer`

```python
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train_cpt(
    model,
    tokenizer,
    dataset: TextDataset,
    output_dir: str,
    *,
    max_steps: int,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    device: str = "cuda",
):
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_steps=max(max_steps // 4, 1),
        save_total_limit=2,
        fp16=(device == "cuda"),
        bf16=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
```

Step count for roughly one epoch:

```python
n_rows = len(dataset)
grad_accum = 8
max_steps = (n_rows + grad_accum - 1) // grad_accum
```

With `batch_size=1` and `grad_accum=8`, each optimizer step sees eight micro-batches.

## 6. Glue it together

```python
def main():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    base = "Qwen/Qwen3-4B-Instruct-2507"  # or a local checkout
    texts = load_cpt_texts(Path("data/domain_cpt.jsonl"))
    # optional smoke: texts = texts[:400]

    model, tokenizer = load_for_cpt(
        base, device, full_lora=(device == "cuda"), lora_r=16 if device == "cuda" else 8
    )
    max_len = 1024 if device == "cuda" else 512
    ds = TextDataset(texts, tokenizer, max_length=max_len)

    grad_accum = 8 if device == "cuda" else 2
    steps = (len(ds) + grad_accum - 1) // grad_accum
    if device != "cuda":
        steps = min(steps, 40)  # smoke only

    train_cpt(
        model,
        tokenizer,
        ds,
        output_dir="out/cpt-lora",
        max_steps=steps,
        grad_accum=grad_accum,
        device=device,
    )

if __name__ == "__main__":
    main()
```

Dependencies:

```bash
pip install "transformers>=4.51" peft accelerate sentencepiece
```

## 7. What “done” looks like after CPT alone

CPT adapters often generate fluent domain-flavored text but messy chat. That is expected. **Part 3** adds a small format SFT on top of this adapter so answers stay concise.

Before that, you can still smoke-generate with a chat template and compare to the base model on fixed factual prompts. If nothing moved on those prompts after a full epoch, check: did you truncate the corpus? Is LoRA too small? Did training actually run on CUDA?

## Checklist

- [ ] JSONL is raw `text`, not chat `messages`
- [ ] Front matter / junk filtered
- [ ] `labels == input_ids` (causal LM)
- [ ] LoRA capacity matches ambition (modules + rank + seq len)
- [ ] Step count ≈ full corpus / effective batch — not “40 steps forever”
- [ ] Adapter saved to durable disk before you kill the machine

## Next

Part 3 will take `out/cpt-lora`, resume it as a trainable PEFT model, and run a short chat SFT so the specialist keeps CPT knowledge while learning to answer cleanly.
