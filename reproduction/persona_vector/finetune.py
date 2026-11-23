"""LoRA fine-tuning on a persona dataset (upstream training.py / sft.py with transformers + peft), optionally with
preventative steering: coef x vector[--steer-layer] added to block layer - 1's output at every position during training.

Each row's `messages` are rendered with the chat template plus the tokenizer's eos token (as upstream), tokenized with
the template's single BOS, cut at MAX_SEQ_LEN, and only assistant turns are trained on (unsloth train_on_responses_only:
tokens after each assistant header up to the next user header). Writes the adapter to --out-dir with train_log.json.

Run from the repo root:
  python -m reproduction.persona_vector.finetune --train-file ... --out-dir ... [--steer-vector ... --steer-layer 16 --steer-coef 1.0]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from reproduction.persona_vector.generate import MODEL
from utils.io import check_counts, read_jsonl, write_meta
from utils.models import load_model_and_tokenizer

# Upstream configs/train_instruct_7b.json.
LORA = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0, "bias": "none", "use_rslora": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
TRAIN = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8, "warmup_steps": 5, "learning_rate": 1e-5,
         "num_train_epochs": 1, "optim": "adamw_bnb_8bit", "weight_decay": 0.01, "lr_scheduler_type": "linear", "seed": 0}
MAX_SEQ_LEN = 2048
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def find(seq: list[int], sub: list[int], start: int) -> int:
    """First index >= start where sub occurs in seq, or -1."""
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


def encode(tokenizer, messages: list[dict]) -> tuple[dict, int]:
    """(input_ids and labels, untruncated length). Labels are -100 except the tokens of each assistant turn (after its
    header, up to the next user header)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token
    full = tokenizer(text, add_special_tokens=False)["input_ids"]  # template already includes BOS
    ids = full[:MAX_SEQ_LEN]
    user = tokenizer(USER_HEADER, add_special_tokens=False)["input_ids"]
    asst = tokenizer(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    i = find(ids, asst, 0)
    while i != -1:
        start = i + len(asst)
        end = find(ids, user, start)
        end = len(ids) if end == -1 else end
        labels[start:end] = ids[start:end]
        i = find(ids, asst, end)
    return {"input_ids": ids, "labels": labels}, len(full)


def add_training_steer(model, vector: np.ndarray, coef: float, block: int):
    """Forward hook adding coef * vector to every position of block `block`'s output (upstream steering_intervention)."""
    v = coef * torch.as_tensor(vector, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)

    def hook(module, inputs, out):
        if isinstance(out, (tuple, list)):
            return (out[0] + v, *out[1:])
        return out + v

    return model.base_model.model.model.layers[block].register_forward_hook(hook)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steer-vector", type=Path, help="npz with response_avg_diff; enables preventative steering.")
    parser.add_argument("--steer-layer", type=int, help="hidden_states index; the hook is on block layer - 1.")
    parser.add_argument("--steer-coef", type=float)
    args = parser.parse_args()
    steer = args.steer_vector is not None
    if steer != (args.steer_layer is not None) or steer != (args.steer_coef is not None):
        raise ValueError("--steer-vector, --steer-layer and --steer-coef go together")
    if (args.out_dir / "adapter_model.safetensors").exists():
        raise FileExistsError(args.out_dir)

    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    rows = read_jsonl(args.train_file)
    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="right")
    encoded = [encode(tokenizer, r["messages"]) for r in rows]
    data = [d for d, _ in encoded]
    n_truncated = sum(n > MAX_SEQ_LEN for _, n in encoded)
    if any(all(l == -100 for l in d["labels"]) for d in data):
        raise AssertionError("a row has no assistant tokens to train on")

    torch.manual_seed(TRAIN["seed"])  # LoRA init
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    handle = None
    if steer:
        vectors = np.load(args.steer_vector)["response_avg_diff"]
        handle = add_training_steer(model, vectors[args.steer_layer], args.steer_coef, args.steer_layer - 1)

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(args.out_dir / "trainer"), bf16=True, logging_steps=1, save_strategy="no",
                               report_to="none", remove_unused_columns=False, **TRAIN),
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    if handle is not None:
        handle.remove()
    model.save_pretrained(args.out_dir)
    (args.out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    write_meta(args.out_dir / "adapter_model.safetensors", [args.train_file] + ([args.steer_vector] if steer else []), args,
               {**check_counts(len(rows), {}, len(data), args.train_file.name), "n_truncated": n_truncated},
               {"model": MODEL, "revision": model.base_model.model.config._commit_hash, "lora": LORA, "train": TRAIN,
                "max_seq_len": MAX_SEQ_LEN, "steer_block": args.steer_layer - 1 if steer else None,
                "steps": trainer.state.global_step, "wall_seconds": time.time() - t0})


if __name__ == "__main__":
    main()
