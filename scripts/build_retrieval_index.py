from __future__ import annotations

import argparse
import json

import torch

from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the index")
    metadata = build_index(
        HPOOntology.from_obo(args.ontology),
        args.model,
        args.output,
        device="cuda",
        batch_size=args.batch_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

