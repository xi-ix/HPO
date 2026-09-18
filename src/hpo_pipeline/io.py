from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)


def locate_entity(document: dict, entity: dict) -> tuple[dict, int] | None:
    start = entity["offset"]
    end = start + entity["length"]
    for section in document["full_text"]:
        local_start = start - section["offset"]
        local_end = end - section["offset"]
        if (
            0 <= local_start <= local_end <= len(section["text"])
            and section["text"][local_start:local_end] == entity["text"]
        ):
            return section, local_start
    return None
