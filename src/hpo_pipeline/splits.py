from __future__ import annotations

import json
from pathlib import Path


def load_split(path: str | Path, documents: list[dict]) -> tuple[list[dict], list[dict]]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    train_ids = set(split["train_pmc_ids"])
    validation_ids = set(split["validation_pmc_ids"])
    document_ids = {document["pmc_id"] for document in documents}
    if train_ids & validation_ids:
        raise ValueError("Train and validation document IDs overlap")
    if train_ids | validation_ids != document_ids:
        raise ValueError("Split document IDs do not exactly match the dataset")
    train = [document for document in documents if document["pmc_id"] in train_ids]
    validation = [document for document in documents if document["pmc_id"] in validation_ids]
    return train, validation

