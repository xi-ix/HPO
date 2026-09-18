from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from hpo_pipeline.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=16)
    args = parser.parse_args()

    identifiers = [document["pmc_id"] for document in read_jsonl(args.input)]
    random.Random(args.seed).shuffle(identifiers)
    validation = sorted(identifiers[: args.validation_size])
    training = sorted(identifiers[args.validation_size :])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"seed": args.seed, "train_pmc_ids": training, "validation_pmc_ids": validation},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(training)} train and {len(validation)} validation ids to {output}")


if __name__ == "__main__":
    main()

