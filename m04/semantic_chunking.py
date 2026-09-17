import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


DATA_PATH = Path("datasets/acme_q3_strategy_memo.txt")
EMBEDDING_MODEL = "text-embedding-3-small"

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
    """
    Split text into sentences without breaking decimal values
    such as 12.4M, 4.2%, or version numbers like v3.2.
    """

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
    """
    Embed every sentence with the same embedding model.
    """

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
    """
    Calculate cosine similarity between two vectors.
    """

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


def semantic_chunk(
    text: str,
    breakpoint_threshold: float = 0.15
) -> list[str]:
    """
    Split text where semantic similarity between
    neighboring sentences drops significantly.
    """

    sentences = split_into_sentences(
        text
    )

    if len(sentences) <= 1:
        return [text]

    embeddings = embed_sentences(
        sentences
    )

    similarities = []

    for index in range(
        len(sentences) - 1
    ):
        similarity = cosine_similarity(
            embeddings[index],
            embeddings[index + 1]
        )

        similarities.append(
            similarity
        )

    average_similarity = float(
        np.mean(similarities)
    )

    print(
        f"\nAverage adjacent similarity: "
        f"{average_similarity:.4f}"
    )

    cutoff = (
        average_similarity
        -
        breakpoint_threshold
    )

    print(
        f"Breakpoint cutoff: "
        f"{cutoff:.4f}"
    )

    print(
        "\nLowest-similarity sentence boundaries:"
    )

    boundary_scores = []

    for index, similarity in enumerate(
        similarities
    ):
        boundary_scores.append(
            (
                similarity,
                index
            )
        )

    boundary_scores.sort()

    for similarity, index in boundary_scores[:8]:

        left = sentences[index]
        right = sentences[index + 1]

        print(
            "\n"
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"LEFT : {left[:100]}"
        )

        print(
            f"RIGHT: {right[:100]}"
        )

        if similarity < cutoff:
            print(">>> BREAKPOINT")

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

    return chunks


def main():

    print("=" * 70)
    print("M04 - SEMANTIC CHUNKING")
    print("=" * 70)

    document = load_document()

    chunks = semantic_chunk(
        document,
        breakpoint_threshold=0.15
    )

    print(
        f"\nCreated {len(chunks)} "
        f"semantic chunks."
    )

    for index, chunk in enumerate(
        chunks
    ):
        print(
            "\n" + "-" * 70
        )

        print(
            f"Chunk {index}"
        )

        print(
            f"Characters: {len(chunk)}"
        )

        print(chunk)


if __name__ == "__main__":
    main()