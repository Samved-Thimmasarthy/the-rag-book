import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


DATA_PATH = Path("datasets/acme_q3_strategy_memo.txt")
EMBEDDING_MODEL = "text-embedding-3-small"

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
]


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def load_document() -> str:
    return DATA_PATH.read_text(
        encoding="utf-8"
    ).strip()


def split_into_sentences(
    text: str
) -> list[str]:

    cleaned = " ".join(
        text.split()
    )

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z"])',
        cleaned
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def embed_sentences(
    sentences: list[str]
) -> np.ndarray:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=sentences
    )

    return np.array(
        [
            item.embedding
            for item in response.data
        ]
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


def calculate_similarities(
    embeddings: np.ndarray
) -> list[float]:

    similarities = []

    for index in range(
        len(embeddings) - 1
    ):
        similarity = cosine_similarity(
            embeddings[index],
            embeddings[index + 1]
        )

        similarities.append(
            similarity
        )

    return similarities


def build_chunks(
    sentences: list[str],
    similarities: list[float],
    threshold: float
) -> tuple[list[str], float]:

    average_similarity = float(
        np.mean(similarities)
    )

    cutoff = (
        average_similarity
        -
        threshold
    )

    breakpoints = []

    for index, similarity in enumerate(
        similarities
    ):
        if similarity < cutoff:
            breakpoints.append(
                index
            )

    chunks = []

    start = 0

    for breakpoint in breakpoints:

        chunk_sentences = sentences[
            start: breakpoint + 1
        ]

        chunks.append(
            " ".join(
                chunk_sentences
            )
        )

        start = breakpoint + 1

    remaining = sentences[
        start:
    ]

    if remaining:
        chunks.append(
            " ".join(
                remaining
            )
        )

    return chunks, cutoff


def main():

    print("=" * 80)
    print("M04 - SEMANTIC CHUNKING THRESHOLD SWEEP")
    print("=" * 80)

    document = load_document()

    sentences = split_into_sentences(
        document
    )

    print(
        f"Sentences detected: "
        f"{len(sentences)}"
    )

    print(
        "\nEmbedding sentences once..."
    )

    embeddings = embed_sentences(
        sentences
    )

    similarities = calculate_similarities(
        embeddings
    )

    average_similarity = float(
        np.mean(similarities)
    )

    print(
        f"Average adjacent similarity: "
        f"{average_similarity:.4f}"
    )

    print(
        "\n"
        f"{'Threshold':<12}"
        f"{'Cutoff':<12}"
        f"{'Chunks':<12}"
        f"{'Avg Size':<12}"
    )

    print("-" * 48)

    for threshold in THRESHOLDS:

        chunks, cutoff = build_chunks(
            sentences,
            similarities,
            threshold
        )

        average_chunk_size = int(
            sum(
                len(chunk)
                for chunk in chunks
            )
            /
            len(chunks)
        )

        print(
            f"{threshold:<12.2f}"
            f"{cutoff:<12.4f}"
            f"{len(chunks):<12}"
            f"{average_chunk_size:<12}"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 80
    )

    print(
        """
Smaller threshold:
- Higher cutoff
- More sentence pairs fall below the cutoff
- More breakpoints
- More, smaller chunks

Larger threshold:
- Lower cutoff
- Fewer sentence pairs fall below the cutoff
- Fewer breakpoints
- Fewer, larger chunks
"""
    )


if __name__ == "__main__":
    main()