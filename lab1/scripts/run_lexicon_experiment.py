from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpo_pipeline.evaluation import evaluate
from hpo_pipeline.io import read_jsonl, write_jsonl
from hpo_pipeline.lexicon import LexiconMatcher
from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.pipeline import PhenotypePipeline
from hpo_pipeline.splits import load_split
from hpo_pipeline.validation import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    documents = list(read_jsonl(args.data))
    training, validation = load_split(args.split, documents)
    ontology = HPOOntology.from_obo(args.ontology)
    pipeline = PhenotypePipeline(LexiconMatcher(ontology, training))
    predictions = [pipeline.predict_document(document)[0] for document in validation]
    errors = validate_submission(validation, predictions, ontology)
    if errors:
        raise ValueError("\n".join(errors))
    result = evaluate(validation, predictions)
    result["experiment"] = {
        "name": "lexicon_baseline",
        "training_documents": len(training),
        "validation_documents": len(validation),
        "prediction_entities": sum(len(document["entities"]) for document in predictions),
    }
    write_jsonl(predictions, args.output)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

