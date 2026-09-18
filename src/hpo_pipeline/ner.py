from __future__ import annotations

from collections import defaultdict

import torch

from .ner_data import LABEL_TO_ID, SectionExample, encode_examples


def predict_spans(
    model,
    tokenizer,
    examples: list[SectionExample],
    device: str,
    max_length: int = 384,
    stride: int = 96,
    batch_size: int = 32,
) -> dict[tuple[str, int], list[tuple[int, int, float]]]:
    rows = encode_examples(tokenizer, examples, max_length=max_length, stride=stride)
    token_scores: dict[tuple[int, int, int], list[torch.Tensor]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            inputs = {
                "input_ids": torch.tensor([row["input_ids"] for row in batch], device=device),
                "attention_mask": torch.tensor(
                    [row["attention_mask"] for row in batch], device=device
                ),
            }
            if batch[0]["token_type_ids"] is not None:
                inputs["token_type_ids"] = torch.tensor(
                    [row["token_type_ids"] for row in batch], device=device
                )
            probabilities = model(**inputs).logits.softmax(dim=-1).cpu()
            for row, row_probabilities in zip(batch, probabilities):
                for offset, probability in zip(row["offset_mapping"], row_probabilities):
                    start, end = offset
                    if start != end:
                        token_scores[(row["example_index"], start, end)].append(probability)

    by_example: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
    for (example_index, start, end), scores in token_scores.items():
        average = torch.stack(scores).mean(dim=0)
        label = int(average.argmax())
        by_example[example_index].append((start, end, label, float(average[label])))

    result: dict[tuple[str, int], list[tuple[int, int, float]]] = {}
    for example_index, example in enumerate(examples):
        spans = []
        current: list[tuple[int, int, float]] = []
        for start, end, label, score in sorted(by_example.get(example_index, [])):
            if label == LABEL_TO_ID["B-PHENO"]:
                if current:
                    spans.append(_finish_span(current, example.text))
                current = [(start, end, score)]
            elif label == LABEL_TO_ID["I-PHENO"]:
                if not current:
                    current = [(start, end, score)]
                else:
                    current.append((start, end, score))
            elif current:
                spans.append(_finish_span(current, example.text))
                current = []
        if current:
            spans.append(_finish_span(current, example.text))
        result[(example.pmc_id, example.section_index)] = [span for span in spans if span[1] > span[0]]
    return result


def _finish_span(tokens: list[tuple[int, int, float]], text: str) -> tuple[int, int, float]:
    start = tokens[0][0]
    end = tokens[-1][1]
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",;:"):
        end -= 1
    confidence = sum(token[2] for token in tokens) / len(tokens)
    return start, end, confidence


def span_metrics(
    examples: list[SectionExample],
    predictions: dict[tuple[str, int], list[tuple[int, int, float]]],
    threshold: float = 0.0,
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    for example in examples:
        gold = set(example.spans)
        predicted = {
            (start, end)
            for start, end, confidence in predictions.get(
                (example.pmc_id, example.section_index), []
            )
            if confidence >= threshold
        }
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

