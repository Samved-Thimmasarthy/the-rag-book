import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_embeddings(
    texts: list[str]
) -> np.ndarray:
    """
    Convert a list of texts into embedding vectors.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
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
    Measure similarity based on vector direction.
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


def show_pair_similarity(
    text_a: str,
    text_b: str
):
    """
    Embed two texts and compare their meaning.
    """

    embeddings = get_embeddings(
        [
            text_a,
            text_b
        ]
    )

    similarity = cosine_similarity(
        embeddings[0],
        embeddings[1]
    )

    print(
        f"\nA: {text_a}"
    )

    print(
        f"B: {text_b}"
    )

    print(
        f"Cosine similarity: "
        f"{similarity:.4f}"
    )


def main():

    print("=" * 70)
    print("M05 - EMBEDDING BASICS")
    print("=" * 70)

    sample_text = (
        "Product Alpha achieved $12.4M "
        "in Q2 revenue."
    )

    embedding = get_embeddings(
        [sample_text]
    )[0]

    print(
        f"\nText: {sample_text}"
    )

    print(
        f"Dimensions: {len(embedding)}"
    )

    print(
        "\nFirst 10 embedding values:"
    )

    for index, value in enumerate(
        embedding[:10]
    ):
        print(
            f"{index:>2}: {value:.6f}"
        )

    print(
        f"\nMinimum value: "
        f"{np.min(embedding):.6f}"
    )

    print(
        f"Maximum value: "
        f"{np.max(embedding):.6f}"
    )

    print(
        f"Mean: "
        f"{np.mean(embedding):.6f}"
    )

    print(
        f"Standard deviation: "
        f"{np.std(embedding):.6f}"
    )

    print(
        f"L2 norm: "
        f"{np.linalg.norm(embedding):.6f}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SEMANTIC SIMILARITY TESTS"
    )

    print(
        "=" * 70
    )

    show_pair_similarity(
        "car",
        "automobile"
    )

    show_pair_similarity(
        "revenue increased this quarter",
        "sales grew during the quarter"
    )

    show_pair_similarity(
        "car",
        "banana"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DANGER ZONE TESTS"
    )

    print(
        "=" * 70
    )

    show_pair_similarity(
        "The product is good.",
        "The product is not good."
    )

    show_pair_similarity(
        "Revenue was $12.4M.",
        "Revenue was $14.2M."
    )

    show_pair_similarity(
        "Alice manages Bob.",
        "Bob manages Alice."
    )


if __name__ == "__main__":
    main()