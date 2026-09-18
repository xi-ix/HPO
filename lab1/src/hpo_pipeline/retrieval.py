from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .ontology import HPOOntology
from .text import normalize_key


def encode_texts(
    texts: list[str], tokenizer, model, device: str, batch_size: int = 256, max_length: int = 64
) -> np.ndarray:
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                output = model(**inputs).last_hidden_state[:, 0]
            output = torch.nn.functional.normalize(output.float(), p=2, dim=1)
            vectors.append(output.cpu().numpy())
    return np.concatenate(vectors).astype("float32")


def build_index(
    ontology: HPOOntology,
    model_path: str,
    output_directory: str | Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict:
    surfaces = []
    seen = set()
    for identifier in sorted(ontology.branch_ids):
        term = ontology.terms[identifier]
        for surface in term.surfaces:
            key = (identifier, normalize_key(surface))
            if len(key[1]) < 2 or key in seen:
                continue
            seen.add(key)
            surfaces.append({"identifier": identifier, "surface": surface})

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    vectors = encode_texts(
        [item["surface"] for item in surfaces], tokenizer, model, device, batch_size=batch_size
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output / "hpo.faiss"))
    with (output / "surfaces.jsonl").open("w", encoding="utf-8") as handle:
        for item in surfaces:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "model": str(Path(model_path).resolve()),
        "root_id": ontology.root_id,
        "terms": len(ontology.branch_ids),
        "surfaces": len(surfaces),
        "dimension": int(vectors.shape[1]),
        "pooling": "CLS",
        "similarity": "cosine",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class DenseRetriever:
    def __init__(self, model_path: str, index_directory: str | Path, device: str = "cuda") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        directory = Path(index_directory)
        self.index = faiss.read_index(str(directory / "hpo.faiss"))
        self.surfaces = [json.loads(line) for line in (directory / "surfaces.jsonl").open()]

    def search(self, mentions: list[str], top_k: int = 10, surface_k: int = 100) -> list[list[dict]]:
        vectors = encode_texts(mentions, self.tokenizer, self.model, self.device)
        return self.search_vectors(vectors, top_k=top_k, surface_k=surface_k)

    def search_vectors(
        self, vectors: np.ndarray, top_k: int = 10, surface_k: int = 100
    ) -> list[list[dict]]:
        scores, indices = self.index.search(vectors, surface_k)
        results = []
        for row_scores, row_indices in zip(scores, indices):
            by_identifier: dict[str, dict] = {}
            for score, index in zip(row_scores, row_indices):
                item = self.surfaces[int(index)]
                previous = by_identifier.get(item["identifier"])
                if previous is None or score > previous["score"]:
                    by_identifier[item["identifier"]] = {
                        "identifier": item["identifier"],
                        "surface": item["surface"],
                        "score": float(score),
                    }
            results.append(
                sorted(by_identifier.values(), key=lambda item: item["score"], reverse=True)[:top_k]
            )
        return results


class TrainingMentionRetriever:
    def __init__(
        self,
        documents: list[dict],
        tokenizer,
        model,
        ontology: HPOOntology,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        pairs = set()
        for document in documents:
            for entity in document.get("entities", []):
                if entity.get("note") == "NO":
                    continue
                for identifier in entity["identifier"].split(";"):
                    canonical_id = ontology.canonical_id(identifier)
                    if canonical_id:
                        pairs.add((entity["text"], canonical_id))
        pairs = sorted(pairs)
        self.items = [{"surface": text, "identifier": identifier} for text, identifier in pairs]
        vectors = encode_texts(
            [item["surface"] for item in self.items],
            tokenizer,
            model,
            device,
            batch_size=batch_size,
        )
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search_vectors(
        self, vectors: np.ndarray, top_k: int = 10, surface_k: int = 100
    ) -> list[list[dict]]:
        scores, indices = self.index.search(vectors, min(surface_k, len(self.items)))
        results = []
        for row_scores, row_indices in zip(scores, indices):
            by_identifier: dict[str, dict] = {}
            for score, index in zip(row_scores, row_indices):
                item = self.items[int(index)]
                previous = by_identifier.get(item["identifier"])
                if previous is None or score > previous["score"]:
                    by_identifier[item["identifier"]] = {
                        "identifier": item["identifier"],
                        "surface": item["surface"],
                        "score": float(score),
                    }
            results.append(
                sorted(by_identifier.values(), key=lambda item: item["score"], reverse=True)[:top_k]
            )
        return results


def fuse_retrieval_candidates(
    ontology_candidates: list[dict],
    training_candidates: list[dict],
    training_weight: float,
    top_k: int = 10,
) -> list[dict]:
    fused: dict[str, dict] = {}
    for candidate in ontology_candidates:
        fused[candidate["identifier"]] = {**candidate, "source": "ontology"}
    for candidate in training_candidates:
        weighted_score = candidate["score"] * training_weight
        current = fused.get(candidate["identifier"])
        if current is None or weighted_score > current["score"]:
            fused[candidate["identifier"]] = {
                **candidate,
                "score": weighted_score,
                "raw_score": candidate["score"],
                "source": "training_mention",
            }
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def _stem_tokens(text: str) -> set[str]:
    irregular = {"ulnae": "ulna", "vertebrae": "vertebra", "teeth": "tooth"}
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        token = irregular.get(token, token)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "is")):
            token = token[:-1]
        tokens.add(token)
    return tokens


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = _stem_tokens(left)
    right_tokens = _stem_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return 2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))


def rerank_candidates(
    mention: str,
    candidates: list[dict],
    ontology: HPOOntology,
    lexical_weight: float,
    depth_weight: float,
) -> list[dict]:
    reranked = []
    for candidate in candidates:
        canonical_id = ontology.canonical_id(candidate["identifier"])
        term = ontology.terms.get(canonical_id) if canonical_id else None
        surfaces = term.surfaces if term else (candidate["surface"],)
        overlap = max(lexical_overlap(mention, surface) for surface in surfaces)
        depth = ontology.depth.get(canonical_id, 0) if canonical_id else 0
        reranked.append(
            {
                **candidate,
                "base_score": candidate["score"],
                "lexical_overlap": overlap,
                "depth": depth,
                "score": candidate["score"]
                + lexical_weight * overlap
                + depth_weight * depth,
            }
        )
    return sorted(reranked, key=lambda item: item["score"], reverse=True)
