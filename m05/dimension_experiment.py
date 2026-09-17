import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"

DIMENSIONS = [
    256,
    512,
    1024,
    1536,
]


TEXTS = [
    "Product Alpha Q2 revenue was $12.4M.",
    "Product Beta churn increased to 4.2%.",
    "Product Gamma launches on September 15.",
    "Dataview is competing with Product Alpha on price.",
    "The Q3 combined revenue target is $18.6M.",
]


QUERIES = [
    (
        "How much revenue did Alpha generate?",
        "$12.4M"
    ),
    (
        "Which product is losing customers?",
        "4.2%"
    ),
    (
        "When is the new AI product launching?",
        "September 15"
    ),
    (
        "Who is threatening Alpha competitively?",
        "Dataview"
    ),
    (
        "What is the total revenue target?",
        "$18.6M"
    ),
]


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def embed(
    texts: list[str],
    dimensions: int
) -> np.ndarray:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=dimensions
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


def storage_per_million(
    dimensions: int
) -> float:

    bytes_required = (
        dimensions
        *
        4
        *
        1_000_000
    )

    return (
        bytes_required
        /
        (1024 * 1024)
    )


def evaluate_dimension(
    dimensions: int
) -> dict:

    text_embeddings = embed(
        TEXTS,
        dimensions
    )

    query_texts = [
        query
        for query, _ in QUERIES
    ]

    query_embeddings = embed(
        query_texts,
        dimensions
    )

    hits = 0
    similarity_total = 0.0
    failures = []

    for index, (
        query,
        expected
    ) in enumerate(QUERIES):

        similarities = []

        for text_embedding in text_embeddings:

            similarity = cosine_similarity(
                query_embeddings[index],
                text_embedding
            )

            similarities.append(
                similarity
            )

        best_index = int(
            np.argmax(similarities)
        )

        best_text = TEXTS[
            best_index
        ]

        best_similarity = similarities[
            best_index
        ]

        similarity_total += (
            best_similarity
        )

        is_correct = (
            expected.lower()
            in
            best_text.lower()
        )

        if is_correct:
            hits += 1
        else:
            failures.append(
                {
                    "query": query,
                    "expected": expected,
                    "retrieved": best_text,
                    "similarity": best_similarity,
                }
            )

    precision = (
        hits
        /
        len(QUERIES)
    )

    average_similarity = (
        similarity_total
        /
        len(QUERIES)
    )

    return {
        "dimensions": dimensions,
        "precision": precision,
        "average_similarity": average_similarity,
        "storage_mb": storage_per_million(
            dimensions
        ),
        "failures": failures,
    }


def main():

    print("=" * 75)
    print("M05 - EMBEDDING DIMENSION EXPERIMENT")
    print("=" * 75)

    results = []

    for dimensions in DIMENSIONS:

        print(
            f"\nTesting "
            f"{dimensions} dimensions..."
        )

        result = evaluate_dimension(
            dimensions
        )

        results.append(
            result
        )

    print(
        "\n" + "=" * 75
    )

    print(
        f"{'Dims':<10}"
        f"{'Precision':<15}"
        f"{'Avg Similarity':<20}"
        f"{'Storage / 1M':<20}"
    )

    print("-" * 65)

    for result in results:

        print(
            f"{result['dimensions']:<10}"
            f"{result['precision']:<15.0%}"
            f"{result['average_similarity']:<20.4f}"
            f"{result['storage_mb']:<10.0f} MB"
        )

    print(
        "\nStorage reduction relative "
        "to 1536 dimensions:"
    )

    full_storage = storage_per_million(
        1536
    )

    for result in results:

        reduction = (
            1
            -
            result["storage_mb"]
            /
            full_storage
        )

        print(
            f"{result['dimensions']:>4} dims: "
            f"{reduction:.1%} less storage"
        )

    print(
        "\n" + "=" * 75
    )

    print(
        "FAILURE ANALYSIS"
    )

    print(
        "=" * 75
    )

    for result in results:

        print(
            f"\n{result['dimensions']} dimensions"
        )

        if not result["failures"]:
            print(
                "No retrieval failures."
            )
            continue

        for failure in result["failures"]:

            print(
                "\nQuery:"
            )

            print(
                failure["query"]
            )

            print(
                "Expected:"
            )

            print(
                failure["expected"]
            )

            print(
                "Retrieved:"
            )

            print(
                failure["retrieved"]
            )

            print(
                f"Similarity: "
                f"{failure['similarity']:.4f}"
            )


if __name__ == "__main__":
    main()