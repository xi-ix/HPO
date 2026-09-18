from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .models import Candidate
from .negation import is_negated
from .ontology import HPOOntology
from .text import inflection_variants, is_word_character, normalize_equal_length, normalize_key


@dataclass
class _TrieNode:
    children: dict[str, _TrieNode] = field(default_factory=dict)
    identifiers: tuple[str, ...] = ()
    surface: str | None = None


class LexiconMatcher:
    def __init__(
        self,
        ontology: HPOOntology,
        training_documents: Iterable[dict] = (),
        min_surface_length: int = 3,
    ) -> None:
        self.ontology = ontology
        self.min_surface_length = min_surface_length
        self.training_frequency: Counter[tuple[str, str]] = Counter()
        surface_to_ids: dict[str, set[str]] = defaultdict(set)
        self.preferred_names: dict[str, str] = {}

        for identifier in ontology.branch_ids:
            term = ontology.terms[identifier]
            self.preferred_names[normalize_key(term.name)] = identifier
            for surface in term.surfaces:
                key = normalize_key(surface)
                if self._valid_surface(key):
                    surface_to_ids[key].add(identifier)
                    for variant in inflection_variants(key):
                        if self._valid_surface(variant):
                            surface_to_ids[variant].add(identifier)

        for document in training_documents:
            for entity in document.get("entities", []):
                key = normalize_key(entity["text"])
                for raw_id in entity["identifier"].split(";"):
                    identifier = ontology.canonical_id(raw_id)
                    if identifier:
                        self.training_frequency[(key, identifier)] += 1
                        if self._valid_surface(key):
                            surface_to_ids[key].add(identifier)
                            for variant in inflection_variants(key):
                                if self._valid_surface(variant):
                                    surface_to_ids[variant].add(identifier)

        self.root = _TrieNode()
        for surface, identifiers in surface_to_ids.items():
            node = self.root
            for character in surface:
                node = node.children.setdefault(character, _TrieNode())
            node.identifiers = tuple(sorted(identifiers))
            node.surface = surface

    def _valid_surface(self, surface: str) -> bool:
        if len(surface) < self.min_surface_length:
            return False
        if not any(character.isalpha() for character in surface):
            return False
        if len(surface) <= 4 and surface.isupper():
            return False
        return surface not in {"all", "normal", "abnormal", "disease", "syndrome", "pain"}

    def find(self, text: str, global_offset: int = 0) -> list[Candidate]:
        normalized = normalize_equal_length(text)
        matches: list[tuple[int, int, _TrieNode]] = []
        for start, character in enumerate(normalized):
            if start and is_word_character(character) and is_word_character(normalized[start - 1]):
                continue
            node = self.root.children.get(character)
            if node is None:
                continue
            cursor = start + 1
            best: tuple[int, _TrieNode] | None = None
            if node.identifiers and self._right_boundary(normalized, cursor):
                best = (cursor, node)
            while cursor < len(normalized):
                node = node.children.get(normalized[cursor])
                if node is None:
                    break
                cursor += 1
                if node.identifiers and self._right_boundary(normalized, cursor):
                    best = (cursor, node)
            if best:
                matches.append((start, best[0], best[1]))

        selected: list[tuple[int, int, _TrieNode]] = []
        for match in sorted(matches, key=lambda value: (value[0], -(value[1] - value[0]))):
            if selected and match[0] < selected[-1][1]:
                continue
            selected.append(match)

        candidates = []
        for start, end, node in selected:
            identifier, score, source = self._rank(node.surface or "", node.identifiers)
            candidates.append(
                Candidate(
                    identifier=identifier,
                    text=text[start:end],
                    offset=global_offset + start,
                    length=end - start,
                    score=score,
                    source=source,
                    negated=is_negated(text, start, end),
                    candidates=node.identifiers,
                )
            )
        return candidates

    @staticmethod
    def _right_boundary(text: str, end: int) -> bool:
        return end == len(text) or not is_word_character(text[end])

    def _rank(self, surface: str, identifiers: tuple[str, ...]) -> tuple[str, float, str]:
        ranked = sorted(
            identifiers,
            key=lambda identifier: (
                self.training_frequency[(surface, identifier)],
                self.preferred_names.get(surface) == identifier,
                self.ontology.depth.get(identifier, 0),
                identifier,
            ),
            reverse=True,
        )
        winner = ranked[0]
        frequency = self.training_frequency[(surface, winner)]
        if frequency:
            return winner, min(0.99, 0.86 + 0.02 * frequency), "training_lexicon"
        if self.preferred_names.get(surface) == winner:
            return winner, 0.9, "hpo_name"
        return winner, 0.78 if len(identifiers) == 1 else 0.62, "hpo_synonym"
