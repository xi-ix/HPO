from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PRF:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(tp, fp, fn, precision, recall, f1)


def _positive_entities(document: dict) -> list[dict]:
    return [entity for entity in document.get("entities", []) if entity.get("note") != "NO"]


def _mention_units(document: dict) -> Counter[tuple]:
    units: Counter[tuple] = Counter()
    for entity in _positive_entities(document):
        identifiers = entity["identifier"].split(";")
        for identifier in identifiers:
            if identifier == "-1":
                units[(entity["offset"], entity["length"], "-1")] += 1
            else:
                units[(entity["offset"], entity["length"], identifier)] += 1
    return units


def evaluate(gold_documents: list[dict], predicted_documents: list[dict]) -> dict:
    predicted_by_id = {document["pmc_id"]: document for document in predicted_documents}
    mention_tp = mention_fp = mention_fn = 0
    document_tp = document_fp = document_fn = 0
    boundary_errors = id_errors = 0

    for gold in gold_documents:
        predicted = predicted_by_id.get(gold["pmc_id"], {"entities": []})
        gold_units = _mention_units(gold)
        predicted_units = _mention_units(predicted)
        overlap = gold_units & predicted_units
        mention_tp += sum(overlap.values())
        mention_fp += sum((predicted_units - gold_units).values())
        mention_fn += sum((gold_units - predicted_units).values())

        gold_ids = {
            identifier
            for entity in _positive_entities(gold)
            for identifier in entity["identifier"].split(";")
            if identifier != "-1"
        }
        predicted_ids = {
            identifier
            for entity in _positive_entities(predicted)
            for identifier in entity["identifier"].split(";")
            if identifier != "-1"
        }
        document_tp += len(gold_ids & predicted_ids)
        document_fp += len(predicted_ids - gold_ids)
        document_fn += len(gold_ids - predicted_ids)

        gold_by_boundary = {
            (entity["offset"], entity["length"]): set(entity["identifier"].split(";"))
            for entity in _positive_entities(gold)
        }
        predicted_by_boundary = {
            (entity["offset"], entity["length"]): set(entity["identifier"].split(";"))
            for entity in _positive_entities(predicted)
        }
        id_errors += sum(
            1
            for boundary in gold_by_boundary.keys() & predicted_by_boundary.keys()
            if not (gold_by_boundary[boundary] & predicted_by_boundary[boundary])
        )
        gold_spans = {(entity["offset"], entity["offset"] + entity["length"]) for entity in _positive_entities(gold)}
        for entity in _positive_entities(predicted):
            span = (entity["offset"], entity["offset"] + entity["length"])
            if span in gold_spans:
                continue
            if any(max(span[0], other[0]) < min(span[1], other[1]) for other in gold_spans):
                boundary_errors += 1

    result = {
        "mention": asdict(_prf(mention_tp, mention_fp, mention_fn)),
        "document": asdict(_prf(document_tp, document_fp, document_fn)),
        "diagnostics": {
            "boundary_overlap_errors": boundary_errors,
            "identifier_errors_at_exact_boundary": id_errors,
            "gold_negated": sum(
                entity.get("note") == "NO" for document in gold_documents for entity in document["entities"]
            ),
            "gold_compound": sum(
                ";" in entity["identifier"] for document in gold_documents for entity in document["entities"]
            ),
            "gold_no_id": sum(
                entity["identifier"] == "-1" for document in gold_documents for entity in document["entities"]
            ),
        },
    }
    return result

