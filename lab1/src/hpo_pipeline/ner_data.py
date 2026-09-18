from __future__ import annotations

from dataclasses import dataclass

LABEL_TO_ID = {"O": 0, "B-PHENO": 1, "I-PHENO": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class SectionExample:
    pmc_id: str
    section_index: int
    offset: int
    text: str
    spans: tuple[tuple[int, int], ...]


def build_section_examples(documents: list[dict]) -> list[SectionExample]:
    examples: list[SectionExample] = []
    for document in documents:
        entities = document.get("entities", [])
        for section_index, section in enumerate(document["full_text"]):
            section_start = section["offset"]
            section_end = section_start + len(section["text"])
            spans = []
            for entity in entities:
                start = entity["offset"]
                end = start + entity["length"]
                if section_start <= start and end <= section_end:
                    spans.append((start - section_start, end - section_start))
            examples.append(
                SectionExample(
                    pmc_id=document["pmc_id"],
                    section_index=section_index,
                    offset=section_start,
                    text=section["text"],
                    spans=tuple(sorted(set(spans))),
                )
            )
    return examples


def encode_examples(tokenizer, examples: list[SectionExample], max_length: int, stride: int):
    rows = []
    for example_index, example in enumerate(examples):
        encoding = tokenizer(
            example.text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            padding="max_length",
        )
        for window_index in range(len(encoding["input_ids"])):
            offsets = encoding["offset_mapping"][window_index]
            labels = []
            for token_start, token_end in offsets:
                if token_start == token_end:
                    labels.append(-100)
                    continue
                containing = next(
                    (
                        (span_start, span_end)
                        for span_start, span_end in example.spans
                        if token_start < span_end and token_end > span_start
                    ),
                    None,
                )
                if containing is None:
                    labels.append(LABEL_TO_ID["O"])
                elif token_start <= containing[0] < token_end:
                    labels.append(LABEL_TO_ID["B-PHENO"])
                else:
                    labels.append(LABEL_TO_ID["I-PHENO"])
            rows.append(
                {
                    "input_ids": encoding["input_ids"][window_index],
                    "attention_mask": encoding["attention_mask"][window_index],
                    "token_type_ids": encoding.get("token_type_ids", [None] * len(encoding["input_ids"]))[
                        window_index
                    ],
                    "labels": labels,
                    "offset_mapping": offsets,
                    "example_index": example_index,
                }
            )
    return rows

