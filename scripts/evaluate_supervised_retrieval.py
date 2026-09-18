from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hpo_pipeline.io import read_jsonl
from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.retrieval import (
    DenseRetriever,
    TrainingMentionRetriever,
    encode_texts,
    fuse_retrieval_candidates,
    rerank_candidates,
)
from hpo_pipeline.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    documents = list(read_jsonl(args.data))
    hpo = HPOOntology.from_obo(args.ontology)
    training, validation = load_split(args.split, documents)
    entities = [
        entity
        for document in validation
        for entity in document["entities"]
        if entity.get("note") != "NO" and entity["identifier"] != "-1"
    ]
    texts = [entity["text"] for entity in entities]
    ontology = DenseRetriever(args.model, args.index, device="cuda")
    mention_retriever = TrainingMentionRetriever(
        training, ontology.tokenizer, ontology.model, hpo, device="cuda"
    )
    vectors = encode_texts(texts, ontology.tokenizer, ontology.model, "cuda")
    ontology_results = ontology.search(texts, top_k=10)
    training_results = mention_retriever.search_vectors(vectors, top_k=10)
    grid = []
    for weight_value in range(80, 111, 5):
        for lexical_weight in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15):
            for depth_weight in (0.0, 0.001, 0.002, 0.005, 0.01):
                weight = weight_value / 100
                hits = total = 0
                single_hits = single_total = 0
                for entity, ontology_candidates, training_candidates in zip(
                    entities, ontology_results, training_results
                ):
                    fused = fuse_retrieval_candidates(
                        ontology_candidates, training_candidates, weight, top_k=10
                    )
                    fused = rerank_candidates(
                        entity["text"], fused, hpo, lexical_weight, depth_weight
                    )
                    predicted = fused[0]["identifier"]
                    gold = entity["identifier"].split(";")
                    total += len(gold)
                    hits += sum(identifier == predicted for identifier in gold)
                    if len(gold) == 1:
                        single_total += 1
                        single_hits += int(predicted == gold[0])
                grid.append(
                    {
                        "training_weight": weight,
                        "lexical_weight": lexical_weight,
                        "depth_weight": depth_weight,
                        "recall_at_1": hits / total,
                        "single_id_top1_accuracy": single_hits / single_total,
                    }
                )
    best = max(grid, key=lambda row: row["recall_at_1"])
    report = {
        "experiment": "sapbert_ontology_plus_training_mentions",
        "training_mention_pairs": len(mention_retriever.items),
        "entities": len(entities),
        "best": best,
        "grid": grid,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "grid"}, indent=2))


if __name__ == "__main__":
    main()
