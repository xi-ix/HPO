from __future__ import annotations

import argparse

import torch
from run_dual_channel_experiment import (
    load_ner_predictions,
    make_candidate_sets,
    select_candidates,
)

from hpo_pipeline.io import read_jsonl, write_jsonl
from hpo_pipeline.lexicon import LexiconMatcher
from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.pipeline import PhenotypePipeline
from hpo_pipeline.retrieval import (
    DenseRetriever,
    TrainingMentionRetriever,
    encode_texts,
    fuse_retrieval_candidates,
    rerank_candidates,
)
from hpo_pipeline.text import normalize_key
from hpo_pipeline.validation import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--ner-predictions", required=True)
    parser.add_argument("--retrieval-model", required=True)
    parser.add_argument("--retrieval-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    documents = list(read_jsonl(args.input))
    training = list(read_jsonl(args.training_data))
    ontology = HPOOntology.from_obo(args.ontology)
    pipeline = PhenotypePipeline(LexiconMatcher(ontology, training))
    supervised_ids = {}
    for document in training:
        for entity in document["entities"]:
            identifiers = [
                ontology.canonical_id(identifier)
                for identifier in entity["identifier"].split(";")
            ]
            identifiers = [identifier for identifier in identifiers if identifier]
            if len(identifiers) == 1:
                supervised_ids.setdefault(normalize_key(entity["text"]), identifiers[0])
    raw = make_candidate_sets(
        documents, pipeline, load_ner_predictions(args.ner_predictions), supervised_ids
    )
    mentions = sorted({item["text"] for items in raw.values() for item in items if not item["lexicon"]})
    retriever = DenseRetriever(args.retrieval_model, args.retrieval_index, device="cuda")
    mention_retriever = TrainingMentionRetriever(
        training, retriever.tokenizer, retriever.model, ontology, device="cuda"
    )
    vectors = encode_texts(mentions, retriever.tokenizer, retriever.model, device="cuda")
    ontology_results = retriever.search_vectors(vectors, top_k=10)
    training_results = mention_retriever.search_vectors(vectors, top_k=10)
    retrieved = [
        rerank_candidates(
            mention,
            fuse_retrieval_candidates(base, supervised, training_weight=1.05, top_k=10),
            ontology,
            lexical_weight=0.1,
            depth_weight=0.01,
        )[:5]
        for mention, base, supervised in zip(mentions, ontology_results, training_results)
    ]
    retrieval_by_text = dict(zip(mentions, retrieved))
    for items in raw.values():
        for item in items:
            candidates = retrieval_by_text.get(item["text"], [])
            item["dense_candidates"] = candidates
            if not item["lexicon"]:
                item["identifier"] = item.get("supervised_id") or (
                    candidates[0]["identifier"] if candidates else None
                )

    predictions = []
    for document in documents:
        selected = select_candidates(
            raw[document["pmc_id"]],
            ner_threshold=0.75,
            replace_threshold=1.0,
            dense_threshold=0.9,
            agreement_threshold=0.95,
            containment_threshold=0.95,
        )
        entities = [
            {
                key: item[key]
                for key in ("identifier", "type", "offset", "length", "text", "note")
            }
            for item in selected
            if item["identifier"] is not None
        ]
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document.get("pmid"),
                "entities": entities,
                "association": [],
            }
        )
    errors = validate_submission(documents, predictions, ontology)
    if errors:
        raise ValueError("\n".join(errors))
    write_jsonl(predictions, args.output)
    print(
        f"wrote {len(predictions)} documents and "
        f"{sum(len(document['entities']) for document in predictions)} entities to {args.output}"
    )


if __name__ == "__main__":
    main()
