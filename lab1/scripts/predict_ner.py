from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from hpo_pipeline.io import read_jsonl, write_jsonl
from hpo_pipeline.ner import predict_spans
from hpo_pipeline.ner_data import build_section_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    examples = build_section_examples(list(read_jsonl(args.input)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model, local_files_only=True).cuda()
    predictions = predict_spans(
        model, tokenizer, examples, device="cuda", batch_size=args.batch_size
    )
    rows = [
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
        for example in examples
    ]
    write_jsonl(rows, args.output)
    print(f"wrote spans for {len(examples)} sections to {args.output}")


if __name__ == "__main__":
    main()

