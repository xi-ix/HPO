from __future__ import annotations

from collections.abc import Iterable

from .lexicon import LexiconMatcher

DEFAULT_EXCLUDED_SECTION_TYPES = frozenset(
    {"REF", "REFERENCES", "ACK_FUND", "SUPPL", "AUTH_CONT", "COMP_INT"}
)
DEFAULT_EXCLUDED_PASSAGE_TYPES = frozenset(
    {
        "ref",
        "reference",
        "references",
        "supplementary-material",
        "acknowledgment",
        "table_caption",
        "table_title_caption",
    }
)


class PhenotypePipeline:
    def __init__(
        self,
        matcher: LexiconMatcher,
        emit_negated: bool = True,
        min_confidence: float = 0.0,
        excluded_section_types: Iterable[str] = DEFAULT_EXCLUDED_SECTION_TYPES,
        excluded_passage_types: Iterable[str] = DEFAULT_EXCLUDED_PASSAGE_TYPES,
    ) -> None:
        self.matcher = matcher
        self.emit_negated = emit_negated
        self.min_confidence = min_confidence
        self.excluded_section_types = {value.upper() for value in excluded_section_types}
        self.excluded_passage_types = {value.lower() for value in excluded_passage_types}

    def predict_document(self, document: dict) -> tuple[dict, list[dict]]:
        candidates = []
        for section in document["full_text"]:
            if section.get("section_type", "").upper() in self.excluded_section_types:
                continue
            if section.get("type", "").lower() in self.excluded_passage_types:
                continue
            candidates.extend(self.matcher.find(section["text"], section["offset"]))

        candidates = [candidate for candidate in candidates if candidate.score >= self.min_confidence]
        unique = {}
        for candidate in candidates:
            key = (candidate.offset, candidate.length, candidate.identifier, candidate.negated)
            previous = unique.get(key)
            if previous is None or candidate.score > previous.score:
                unique[key] = candidate
        ordered = sorted(unique.values(), key=lambda value: (value.offset, value.length, value.identifier))
        entities = [
            candidate.as_entity(emit_negated=self.emit_negated)
            for candidate in ordered
            if self.emit_negated or not candidate.negated
        ]
        traces = [
            {
                "pmc_id": document["pmc_id"],
                "identifier": candidate.identifier,
                "offset": candidate.offset,
                "length": candidate.length,
                "text": candidate.text,
                "score": round(candidate.score, 4),
                "source": candidate.source,
                "negated": candidate.negated,
                "candidates": list(candidate.candidates),
            }
            for candidate in ordered
        ]
        output = {
            "pmc_id": document["pmc_id"],
            "pmid": document.get("pmid"),
            "entities": entities,
            "association": [],
        }
        return output, traces
