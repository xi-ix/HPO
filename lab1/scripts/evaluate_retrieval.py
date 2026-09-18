from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hpo_pipeline.io import read_jsonl
from hpo_pipeline.retrieval import DenseRetriever
from hpo_pipeline.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for retrieval evaluation")

    documents = list(read_jsonl(args.data))
    _, validation = load_split(args.split, documents)
    entities = [
        entity
        for document in validation
        for entity in document["entities"]
        if entity.get("note") != "NO" and entity["identifier"] != "-1"
    ]
    retriever = DenseRetriever(args.model, args.index, device="cuda")
    retrieved = retriever.search([entity["text"] for entity in entities], top_k=10)
    unit_total = 0
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal_rank = 0.0
    single_top1 = single_total = 0
    errors = []
    for entity, candidates in zip(entities, retrieved):
        gold_ids = [identifier for identifier in entity["identifier"].split(";") if identifier != "-1"]
        candidate_ids = [candidate["identifier"] for candidate in candidates]
        for identifier in gold_ids:
            unit_total += 1
            rank = candidate_ids.index(identifier) + 1 if identifier in candidate_ids else None
            for k in hits:
                hits[k] += int(rank is not None and rank <= k)
            reciprocal_rank += 1 / rank if rank else 0.0
        if len(gold_ids) == 1:
            single_total += 1
            single_top1 += int(bool(candidate_ids) and candidate_ids[0] == gold_ids[0])
        if not set(gold_ids) & set(candidate_ids[:5]):
            errors.append(
                {
                    "text": entity["text"],
                    "gold": gold_ids,
                    "top5": candidates[:5],
                }
            )
    report = {
        "experiment": "sapbert_gold_mention_retrieval",
        "entities": len(entities),
        "hpo_units": unit_total,
        "recall_at_1": hits[1] / unit_total,
        "recall_at_5": hits[5] / unit_total,
        "recall_at_10": hits[10] / unit_total,
        "mean_reciprocal_rank_at_10": reciprocal_rank / unit_total,
        "single_id_top1_accuracy": single_top1 / single_total,
        "single_id_entities": single_total,
        "top5_miss_examples": errors[:100],
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "top5_miss_examples"}, indent=2))


if __name__ == "__main__":
    main()

