from __future__ import annotations

from collections import Counter

from .io import locate_entity
from .ontology import HPOOntology


def validate_submission(
    source_documents: list[dict], predictions: list[dict], ontology: HPOOntology
) -> list[str]:
    errors: list[str] = []
    source_by_id = {document["pmc_id"]: document for document in source_documents}
    predicted_ids = [document.get("pmc_id") for document in predictions]
    for pmc_id, count in Counter(predicted_ids).items():
        if count > 1:
            errors.append(f"duplicate document: {pmc_id}")
    missing = set(source_by_id) - set(predicted_ids)
    extra = set(predicted_ids) - set(source_by_id)
    if missing:
        errors.append(f"missing documents: {sorted(missing)}")
    if extra:
        errors.append(f"extra documents: {sorted(extra)}")

    required = {"identifier", "type", "offset", "length", "text", "note"}
    for prediction in predictions:
        pmc_id = prediction.get("pmc_id")
        source = source_by_id.get(pmc_id)
        if source is None:
            continue
        if prediction.get("pmid") != source.get("pmid"):
            errors.append(f"{pmc_id}: pmid differs from input")
        seen = set()
        for index, entity in enumerate(prediction.get("entities", [])):
            prefix = f"{pmc_id}: entity {index}"
            absent = required - set(entity)
            if absent:
                errors.append(f"{prefix}: missing fields {sorted(absent)}")
                continue
            key = tuple(entity.get(name) for name in ("identifier", "offset", "length", "text", "note"))
            if key in seen:
                errors.append(f"{prefix}: duplicate entity")
            seen.add(key)
            if entity["type"] != "Phenotype":
                errors.append(f"{prefix}: type must be Phenotype")
            if entity["note"] not in (None, "NO"):
                errors.append(f"{prefix}: note must be null or NO")
            if not isinstance(entity["offset"], int) or not isinstance(entity["length"], int):
                errors.append(f"{prefix}: offset and length must be integers")
            elif locate_entity(source, entity) is None:
                errors.append(f"{prefix}: coordinates do not reproduce text")
            if not ontology.validate_identifier(entity["identifier"]):
                errors.append(f"{prefix}: identifier outside HP:0000118 branch: {entity['identifier']}")
    return errors

