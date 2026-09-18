from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from hpo_pipeline.io import read_jsonl, write_jsonl
from hpo_pipeline.ner import predict_spans, span_metrics
from hpo_pipeline.ner_data import build_section_examples
from hpo_pipeline.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=96)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NER evaluation")

    documents = list(read_jsonl(args.data))
    _, validation = load_split(args.split, documents)
    examples = build_section_examples(validation)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model, local_files_only=True).cuda()
    predictions = predict_spans(
        model,
        tokenizer,
        examples,
        device="cuda",
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    thresholds = [round(value / 100, 2) for value in range(0, 100, 5)]
    metrics = {str(threshold): span_metrics(examples, predictions, threshold) for threshold in thresholds}
    best_threshold = max(thresholds, key=lambda threshold: metrics[str(threshold)]["f1"])
    report = {
        "experiment": "biobert_ner_threshold",
        "best_threshold": best_threshold,
        "best": metrics[str(best_threshold)],
        "thresholds": metrics,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for example in examples:
        rows.append(
            {
                "pmc_id": example.pmc_id,
                "section_index": example.section_index,
                "section_offset": example.offset,
                "spans": [
                    {"start": start, "end": end, "confidence": confidence}
                    for start, end, confidence in predictions.get(
                        (example.pmc_id, example.section_index), []
                    )
                ],
            }
        )
    write_jsonl(rows, args.predictions)
    print(json.dumps({key: value for key, value in report.items() if key != "thresholds"}, indent=2))


if __name__ == "__main__":
    main()

