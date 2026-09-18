from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

from .models import HPOTerm

_SYNONYM_RE = re.compile(r'^synonym: "((?:[^"\\]|\\.)*)"')
_DEF_RE = re.compile(r'^def: "((?:[^"\\]|\\.)*)"')


class HPOOntology:
    def __init__(self, terms: dict[str, HPOTerm], root_id: str = "HP:0000118") -> None:
        self.terms = terms
        self.root_id = root_id
        self.alt_to_primary = {
            alt_id: term.identifier for term in terms.values() for alt_id in term.alt_ids
        }
        children: dict[str, set[str]] = defaultdict(set)
        for term in terms.values():
            for parent in term.parents:
                children[parent].add(term.identifier)
        self.children = {key: frozenset(value) for key, value in children.items()}
        self.branch_ids = self._descendants(root_id)
        self.depth = self._compute_depths(root_id)

    @classmethod
    def from_obo(cls, path: str | Path, root_id: str = "HP:0000118") -> HPOOntology:
        terms: dict[str, HPOTerm] = {}
        current: dict[str, object] | None = None

        def commit() -> None:
            nonlocal current
            if not current or current.get("obsolete") or "id" not in current or "name" not in current:
                current = None
                return
            identifier = str(current["id"])
            terms[identifier] = HPOTerm(
                identifier=identifier,
                name=str(current["name"]),
                synonyms=tuple(current.get("synonyms", [])),
                parents=tuple(current.get("parents", [])),
                alt_ids=tuple(current.get("alt_ids", [])),
                definition=current.get("definition"),
            )
            current = None

        with Path(path).open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line == "[Term]":
                    commit()
                    current = {"synonyms": [], "parents": [], "alt_ids": []}
                elif line.startswith("["):
                    commit()
                elif current is None:
                    continue
                elif line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("alt_id: "):
                    current["alt_ids"].append(line[8:])
                elif line.startswith("is_a: "):
                    current["parents"].append(line[6:].split(" ! ", 1)[0])
                elif line.startswith("synonym: "):
                    match = _SYNONYM_RE.match(line)
                    if match:
                        current["synonyms"].append(_unescape(match.group(1)))
                elif line.startswith("def: "):
                    match = _DEF_RE.match(line)
                    if match:
                        current["definition"] = _unescape(match.group(1))
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
            commit()
        if root_id not in terms:
            raise ValueError(f"HPO root {root_id} not found in {path}")
        return cls(terms, root_id=root_id)

    def _descendants(self, root_id: str) -> frozenset[str]:
        seen: set[str] = set()
        queue = deque([root_id])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(self.children.get(node, ()))
        return frozenset(seen)

    def _compute_depths(self, root_id: str) -> dict[str, int]:
        depth = {root_id: 0}
        queue = deque([root_id])
        while queue:
            node = queue.popleft()
            for child in self.children.get(node, ()):
                proposed = depth[node] + 1
                if proposed > depth.get(child, -1):
                    depth[child] = proposed
                    queue.append(child)
        return depth

    def canonical_id(self, identifier: str) -> str | None:
        identifier = self.alt_to_primary.get(identifier, identifier)
        return identifier if identifier in self.branch_ids else None

    def validate_identifier(self, identifier: str) -> bool:
        return identifier == "-1" or all(
            part == "-1" or self.canonical_id(part) is not None
            for part in identifier.split(";")
        )


def _unescape(value: str) -> str:
    return value.replace(r'\"', '"').replace(r"\\", "\\")

