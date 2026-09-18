from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from hpo_pipeline.evaluation import evaluate
from hpo_pipeline.io import read_jsonl, write_jsonl
from hpo_pipeline.lexicon import LexiconMatcher
from hpo_pipeline.negation import is_negated
from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.pipeline import DEFAULT_EXCLUDED_PASSAGE_TYPES, PhenotypePipeline
from hpo_pipeline.retrieval import (
    DenseRetriever,
    TrainingMentionRetriever,
    encode_texts,
    fuse_retrieval_candidates,
    rerank_candidates,
)
from hpo_pipeline.splits import load_split
from hpo_pipeline.text import normalize_key
from hpo_pipeline.validation import validate_submission


def overlaps(left: dict, right: dict) -> bool:
    return max(left["offset"], right["offset"]) < min(
        left["offset"] + left["length"], right["offset"] + right["length"]
    )


def load_ner_predictions(path: str) -> dict[tuple[str, int], list[dict]]:
    return {
        (row["pmc_id"], row["section_index"]): row["spans"] for row in read_jsonl(path)
    }


def make_candidate_sets(
    validation: list[dict],
    lexicon_pipeline: PhenotypePipeline,
    ner_by_section: dict,
    supervised_ids: dict[str, str],
) -> dict[str, list[dict]]:
    result = {}
    for document in validation:
        lexicon_prediction, _ = lexicon_pipeline.predict_document(document)
        lexicon = [
            {
                **entity,
                "lexicon_id": entity["identifier"],
                "lexicon": True,
                "ner_confidence": None,
            }
            for entity in lexicon_prediction["entities"]
        ]
        ner = []
        for section_index, section in enumerate(document["full_text"]):
            if section.get("type", "").lower() in DEFAULT_EXCLUDED_PASSAGE_TYPES:
                continue
            for span in ner_by_section.get((document["pmc_id"], section_index), []):
                start, end = span["start"], span["end"]
                ner.append(
                    {
                        "identifier": None,
                        "type": "Phenotype",
                        "offset": section["offset"] + start,
                        "length": end - start,
                        "text": section["text"][start:end],
                        "note": "NO" if is_negated(section["text"], start, end) else None,
                        "lexicon_id": None,
                        "lexicon": False,
                        "ner_confidence": span["confidence"],
                        "supervised_id": supervised_ids.get(normalize_key(section["text"][start:end])),
                    }
                )
        exact = {(item["offset"], item["length"]): item for item in lexicon}
        for item in ner:
            key = (item["offset"], item["length"])
            if key in exact:
                exact[key]["ner_confidence"] = item["ner_confidence"]
            else:
                lexicon.append(item)
        result[document["pmc_id"]] = lexicon
    return result


def select_candidates(
    raw: list[dict],
    ner_threshold: float,
    replace_threshold: float,
    dense_threshold: float,
    agreement_threshold: float,
    containment_threshold: float,
) -> list[dict]:
    selected = [item.copy() for item in raw if item["lexicon"]]
    ner_only = sorted(
        (
            item
            for item in raw
            if not item["lexicon"] and item["ner_confidence"] >= ner_threshold
            and (
                item.get("supervised_id") is not None
                or (
                    item.get("dense_candidates")
                    and item["dense_candidates"][0].get(
                        "raw_score", item["dense_candidates"][0].get("base_score", 0.0)
                    )
                    >= dense_threshold
                )
            )
        ),
        key=lambda item: item["ner_confidence"],
        reverse=True,
    )
    for item in ner_only:
        conflicts = [current for current in selected if overlaps(item, current)]
        if conflicts:
            item_end = item["offset"] + item["length"]
            strict_contains = len(conflicts) == 1 and (
                item["offset"] <= conflicts[0]["offset"]
                and item_end >= conflicts[0]["offset"] + conflicts[0]["length"]
                and (
                    item["offset"] < conflicts[0]["offset"]
                    or item_end > conflicts[0]["offset"] + conflicts[0]["length"]
                )
            )
            mapping_agrees = all(
                current["identifier"] == item["identifier"] for current in conflicts
            )
            may_replace = item["ner_confidence"] >= replace_threshold or (
                mapping_agrees and item["ner_confidence"] >= agreement_threshold
            ) or (
                strict_contains
                and item["ner_confidence"] >= containment_threshold
            )
            if not may_replace:
                continue
        if conflicts:
            selected = [current for current in selected if current not in conflicts]
        if not any(overlaps(item, current) for current in selected):
            selected.append(item.copy())
    return sorted(selected, key=lambda item: (item["offset"], item["length"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ner-predictions", required=True)
    parser.add_argument("--retrieval-model", required=True)
    parser.add_argument("--retrieval-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    documents = list(read_jsonl(args.data))
    training, validation = load_split(args.split, documents)
    ontology = HPOOntology.from_obo(args.ontology)
    lexicon_pipeline = PhenotypePipeline(LexiconMatcher(ontology, training))
    supervised_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for document in training:
        for entity in document["entities"]:
            identifiers = [
                ontology.canonical_id(identifier)
                for identifier in entity["identifier"].split(";")
            ]
            identifiers = [identifier for identifier in identifiers if identifier]
            if len(identifiers) == 1:
                supervised_counts[normalize_key(entity["text"])][identifiers[0]] += 1
    supervised_ids = {
        text: counts.most_common(1)[0][0] for text, counts in supervised_counts.items()
    }
    raw = make_candidate_sets(
        validation,
        lexicon_pipeline,
        load_ner_predictions(args.ner_predictions),
        supervised_ids,
    )
    unique_mentions = sorted(
        {item["text"] for items in raw.values() for item in items if not item["lexicon"]}
    )
    retriever = DenseRetriever(args.retrieval_model, args.retrieval_index, device="cuda")
    mention_retriever = TrainingMentionRetriever(
        training, retriever.tokenizer, retriever.model, ontology, device="cuda"
    )
    mention_vectors = encode_texts(
        unique_mentions, retriever.tokenizer, retriever.model, device="cuda"
    )
    ontology_retrieved = retriever.search_vectors(mention_vectors, top_k=10)
    training_retrieved = mention_retriever.search_vectors(mention_vectors, top_k=10)
    retrieved = [
        rerank_candidates(
            mention,
            fuse_retrieval_candidates(
                ontology_candidates, supervised, training_weight=1.05, top_k=10
            ),
            ontology,
            lexical_weight=0.1,
            depth_weight=0.01,
        )[:5]
        for mention, ontology_candidates, supervised in zip(
            unique_mentions, ontology_retrieved, training_retrieved
        )
    ]
    retrieval_by_text = dict(zip(unique_mentions, retrieved))
    for items in raw.values():
        for item in items:
            candidates = retrieval_by_text.get(item["text"], [])
            item["dense_candidates"] = candidates
            if not item["lexicon"]:
                item["identifier"] = item.get("supervised_id") or (
                    candidates[0]["identifier"] if candidates else None
                )

    grid = []
    best = None
    thresholds = [0.65, 0.7, 0.75, 0.8, 0.85]
    replace_thresholds = [1.0]
    dense_thresholds = [0.85, 0.9, 0.95]
    agreement_thresholds = [0.9, 0.95, 1.0]
    containment_thresholds = [round(value / 100, 2) for value in range(80, 101, 5)]
    for ner_threshold in thresholds:
        for replace_threshold in replace_thresholds:
            for dense_threshold in dense_thresholds:
                for agreement_threshold in agreement_thresholds:
                    for containment_threshold in containment_thresholds:
                        predictions = []
                        for document in validation:
                            selected = select_candidates(
                                raw[document["pmc_id"]],
                                ner_threshold,
                                replace_threshold,
                                dense_threshold,
                                agreement_threshold,
                                containment_threshold,
                            )
                            entities = []
                            for item in selected:
                                if item["identifier"] is None:
                                    continue
                                entities.append(
                                    {
                                        key: item[key]
                                        for key in (
                                            "identifier",
                                            "type",
                                            "offset",
                                            "length",
                                            "text",
                                            "note",
                                        )
                                    }
                                )
                            predictions.append(
                                {
                                    "pmc_id": document["pmc_id"],
                                    "pmid": document.get("pmid"),
                                    "entities": entities,
                                    "association": [],
                                }
                            )
                        metrics = evaluate(validation, predictions)
                        record = {
                            "ner_threshold": ner_threshold,
                            "replace_threshold": replace_threshold,
                            "dense_threshold": dense_threshold,
                            "agreement_threshold": agreement_threshold,
                            "containment_threshold": containment_threshold,
                            "mention_f1": metrics["mention"]["f1"],
                            "document_f1": metrics["document"]["f1"],
                            "predictions": predictions,
                            "metrics": metrics,
                        }
                        grid.append(
                            {
                                key: value
                                for key, value in record.items()
                                if key not in {"predictions", "metrics"}
                            }
                        )
                        if best is None or record["mention_f1"] > best["mention_f1"]:
                            best = record

    assert best is not None
    errors = validate_submission(validation, best["predictions"], ontology)
    if errors:
        raise ValueError("\n".join(errors))
    write_jsonl(best["predictions"], args.output)
    report = {
        "experiment": "dual_channel_lexicon_biobert_sapbert",
        "best_parameters": {
            "ner_threshold": best["ner_threshold"],
            "replace_threshold": best["replace_threshold"],
            "dense_threshold": best["dense_threshold"],
            "agreement_threshold": best["agreement_threshold"],
            "containment_threshold": best["containment_threshold"],
        },
        "best_metrics": best["metrics"],
        "grid": grid,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "grid"}, indent=2))


if __name__ == "__main__":
    main()
