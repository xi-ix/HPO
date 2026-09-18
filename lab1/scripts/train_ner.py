from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from hpo_pipeline.io import read_jsonl
from hpo_pipeline.ner import predict_spans, span_metrics
from hpo_pipeline.ner_data import ID_TO_LABEL, LABEL_TO_ID, build_section_examples, encode_examples
from hpo_pipeline.splits import load_split


class TokenDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        result = {
            "input_ids": torch.tensor(row["input_ids"]),
            "attention_mask": torch.tensor(row["attention_mask"]),
            "labels": torch.tensor(row["labels"]),
        }
        if row["token_type_ids"] is not None:
            result["token_type_ids"] = torch.tensor(row["token_type_ids"])
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--train-all", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    device = "cuda"
    documents = list(read_jsonl(args.data))
    if args.train_all:
        train_documents, validation_documents = documents, []
    else:
        train_documents, validation_documents = load_split(args.split, documents)
    train_examples = build_section_examples(train_documents)
    validation_examples = build_section_examples(validation_documents)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    train_rows = encode_examples(tokenizer, train_examples, args.max_length, args.stride)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        local_files_only=True,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    loader = DataLoader(
        TokenDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=total_steps
    )
    label_counts = torch.zeros(len(LABEL_TO_ID), dtype=torch.long)
    for row in train_rows:
        for label in row["labels"]:
            if label >= 0:
                label_counts[label] += 1
    weights = (label_counts.sum() / label_counts.clamp_min(1)).float().pow(
        args.class_weight_power
    ).to(device)
    weights = weights / weights.mean()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits
                loss = F.cross_entropy(
                    logits.view(-1, len(LABEL_TO_ID)),
                    labels.view(-1),
                    weight=weights,
                    ignore_index=-100,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss)

        if validation_examples:
            predictions = predict_spans(
                model,
                tokenizer,
                validation_examples,
                device=device,
                max_length=args.max_length,
                stride=args.stride,
                batch_size=args.batch_size * 2,
            )
            metrics = span_metrics(validation_examples, predictions)
        else:
            metrics = {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        record = {"epoch": epoch, "loss": running_loss / len(loader), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if not validation_examples or metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "best_metrics.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )

    run = {
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_windows": len(train_rows),
        "label_counts": label_counts.tolist(),
        "class_weights": weights.cpu().tolist(),
        "class_weight_power": args.class_weight_power,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "stride": args.stride,
        "history": history,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
