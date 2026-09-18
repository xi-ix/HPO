from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hpo_pipeline.evaluation import evaluate
from hpo_pipeline.lexicon import LexiconMatcher
from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.text import inflection_variants

OBO = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0001000
name: Test phenotype
synonym: "Phenotype finding" EXACT []
is_a: HP:0000118 ! Phenotypic abnormality
"""


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "hp.obo"
        path.write_text(OBO, encoding="utf-8")
        self.ontology = HPOOntology.from_obo(path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_branch_and_matching_preserve_offsets(self) -> None:
        matcher = LexiconMatcher(self.ontology)
        text = "No test phenotype, but phenotype finding was present."
        matches = matcher.find(text, global_offset=100)
        self.assertEqual([item.text for item in matches], ["test phenotype", "phenotype finding"])
        self.assertEqual(matches[0].offset, 103)
        self.assertTrue(matches[0].negated)
        self.assertFalse(matches[1].negated)

    def test_evaluation_excludes_negated(self) -> None:
        gold = [{"pmc_id": "1", "entities": [
            {"identifier": "HP:0001000", "offset": 3, "length": 4, "note": None},
            {"identifier": "HP:0001000", "offset": 9, "length": 4, "note": "NO"},
        ]}]
        prediction = [{"pmc_id": "1", "entities": [
            {"identifier": "HP:0001000", "offset": 3, "length": 4, "note": None}
        ]}]
        result = evaluate(gold, prediction)
        self.assertEqual(result["mention"]["f1"], 1.0)
        self.assertEqual(result["document"]["f1"], 1.0)

    def test_conservative_inflection_variants(self) -> None:
        self.assertEqual(inflection_variants("developmental delay"), {"developmental delays"})
        self.assertEqual(inflection_variants("ataxias"), {"ataxia"})
        self.assertEqual(inflection_variants("visual loss"), set())


if __name__ == "__main__":
    unittest.main()
