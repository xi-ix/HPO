from __future__ import annotations

import argparse
import json
import sys

from .evaluation import evaluate
from .io import read_jsonl, write_jsonl
from .lexicon import LexiconMatcher
from .ontology import HPOOntology
from .pipeline import PhenotypePipeline
from .validation import validate_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatientPheX task 1 pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-ontology", help="summarize the HPO branch")
    inspect_parser.add_argument("--ontology", required=True)

    predict_parser = subparsers.add_parser("predict", help="create task 1 predictions")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--ontology", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--training-data")
    predict_parser.add_argument("--trace-output")
    predict_parser.add_argument("--min-confidence", type=float, default=0.0)
    predict_parser.add_argument("--drop-negated", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate a submission")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--prediction", required=True)
    validate_parser.add_argument("--ontology", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate labeled JSONL")
    evaluate_parser.add_argument("--gold", required=True)
    evaluate_parser.add_argument("--prediction", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "inspect-ontology":
        ontology = HPOOntology.from_obo(arguments.ontology)
        print(json.dumps({"terms": len(ontology.terms), "branch_terms": len(ontology.branch_ids)}))
        return 0
    if arguments.command == "predict":
        ontology = HPOOntology.from_obo(arguments.ontology)
        training = list(read_jsonl(arguments.training_data)) if arguments.training_data else []
        matcher = LexiconMatcher(ontology, training)
        pipeline = PhenotypePipeline(
            matcher,
            emit_negated=not arguments.drop_negated,
            min_confidence=arguments.min_confidence,
        )
        predictions = []
        traces = []
        for document in read_jsonl(arguments.input):
            prediction, document_traces = pipeline.predict_document(document)
            predictions.append(prediction)
            traces.extend(document_traces)
        write_jsonl(predictions, arguments.output)
        if arguments.trace_output:
            write_jsonl(traces, arguments.trace_output)
        print(f"wrote {len(predictions)} documents to {arguments.output}")
        return 0
    if arguments.command == "validate":
        ontology = HPOOntology.from_obo(arguments.ontology)
        errors = validate_submission(
            list(read_jsonl(arguments.input)), list(read_jsonl(arguments.prediction)), ontology
        )
        if errors:
            print("\n".join(errors), file=sys.stderr)
            print(f"validation failed with {len(errors)} error(s)", file=sys.stderr)
            return 1
        print("validation passed")
        return 0
    if arguments.command == "evaluate":
        result = evaluate(
            list(read_jsonl(arguments.gold)), list(read_jsonl(arguments.prediction))
        )
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(main())

