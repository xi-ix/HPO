from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HPOTerm:
    identifier: str
    name: str
    synonyms: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    alt_ids: tuple[str, ...] = ()
    definition: str | None = None

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.name, *self.synonyms)))


@dataclass(frozen=True)
class Candidate:
    identifier: str
    text: str
    offset: int
    length: int
    score: float
    source: str
    negated: bool = False
    candidates: tuple[str, ...] = field(default_factory=tuple)

    def as_entity(self, emit_negated: bool = True) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "type": "Phenotype",
            "offset": self.offset,
            "length": self.length,
            "text": self.text,
            "note": "NO" if self.negated and emit_negated else None,
        }

