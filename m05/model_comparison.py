import os
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


CHUNKS = [
    (
        "Product Alpha achieved Q2 revenue of $12.4M. "
        "Current ARR reached $48.2M."
    ),
    (
        "Product Beta generated $3.8M in Q2 revenue. "
        "Churn increased to 4.2%. "
        "Customer retention remains a concern."
    ),
    (
        "Product Gamma remains in private beta. "
        "Launch is scheduled for September 15."
    ),
    (
        "Combined Q3 revenue target is $18.6M. "
        "Product Alpha target is $14.2M, "
        "Product Beta target is $4.0M, "
        "and Product Gamma target is $0.4M."
    ),
    (
        "Risk: Competitive threat from Nexus Corp's "
        "new enterprise offering. "
        "The threat could affect Product Alpha's market position."
    ),
    (
        "Risk: Beta churn exceeding the 5% threshold. "
        "Mitigation: dedicated customer success support "
        "for at-risk accounts."
    ),
    (
        "Risk: Gamma launch delay beyond September 30. "
        "Mitigation: weekly launch-readiness reviews."
    ),
    (
        "Risk: Key person dependency on Alpha's "
        "machine learning infrastructure team."
    ),
    (
        "Risk: Regulatory changes in the EU could affect "
        "data processing capabilities."
    ),
]


QUERIES = [
    {
        "query": "What is Product Alpha's revenue?",
        "expected": "$12.4M",
        "type": "factual",
    },
    {
        "query": "What is Beta's churn rate?",
        "expected": "4.2%",
        "type": "factual",
    },
    {
        "query": "Which product is losing customers fastest?",
        "expected": "4.2%",
        "type": "semantic",
    },
    {
        "query": "What new product is coming soon?",
        "expected": "September 15",
        "type": "semantic",
    },
    {
        "query": "What threatens Alpha's market position?",
        "expected": "Nexus Corp",
        "type": "analytical",
    },
    {
        "query": "How is the company reducing customer attrition?",
        "expected": "customer success",
        "type": "analytical",
    },
    {
        "query": "Tell me about Nexus Corp.",
        "expected": "Nexus Corp",
        "type": "entity",
    },
    {
        "query": "What is the EU risk?",
        "expected": "Regulatory changes",
        "type": "entity",
    },
]


MODELS = [
    {
        "name": "text-embedding-3-small",
        "dimensions": 1536,
    },
    {
        "name": "text-embedding-3-large",
        "dimensions": 3072,
    },
    {
        "name": "text-embedding-ada-002",
        "dimensions": 1536,
    },
]


def embed(
    texts: list[str],
    model: str
) -> tuple[np.ndarray, float]:
    """
    Embed a batch of texts and measure API latency.
    """

    start = time.perf_counter()

    response = client.embeddings.create(
        model=model,
        input=texts
    )

    latency = (
        time.perf_counter()
        -
        start
    )

    embeddings = np.array(
        [
            item.embedding
            for item in response.data
        ]
    )

    return embeddings, latency


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


def evaluate_model(
    model_info: dict
) -> dict:

    model_name = model_info[
        "name"
    ]

    chunk_embeddings, chunk_latency = embed(
        CHUNKS,
        model_name
    )

    query_texts = [
        item["query"]
        for item in QUERIES
    ]

    query_embeddings, query_latency = embed(
        query_texts,
        model_name
    )

    hits = 0
    query_results = []

    for index, query_info in enumerate(
        QUERIES
    ):

        similarities = []

        for chunk_embedding in chunk_embeddings:

            similarity = cosine_similarity(
                query_embeddings[index],
                chunk_embedding
            )

            similarities.append(
                similarity
            )

        best_index = int(
            np.argmax(similarities)
        )

        best_chunk = CHUNKS[
            best_index
        ]

        best_similarity = similarities[
            best_index
        ]

        found = (
            query_info["expected"].lower()
            in
            best_chunk.lower()
        )

        if found:
            hits += 1

        query_results.append(
            {
                "query": query_info["query"],
                "type": query_info["type"],
                "expected": query_info["expected"],
                "retrieved": best_chunk,
                "similarity": best_similarity,
                "correct": found,
            }
        )

    return {
        "model": model_name,
        "dimensions": len(
            chunk_embeddings[0]
        ),
        "hits": hits,
        "total": len(QUERIES),
        "precision": (
            hits
            /
            len(QUERIES)
        ),
        "chunk_latency": chunk_latency,
        "query_latency": query_latency,
        "results": query_results,
    }


def main():

    print("=" * 80)
    print("M05 - EMBEDDING MODEL COMPARISON")
    print("=" * 80)

    all_results = []

    for model_info in MODELS:

        model_name = model_info[
            "name"
        ]

        print(
            f"\nTesting: {model_name}"
        )

        try:

            result = evaluate_model(
                model_info
            )

            all_results.append(
                result
            )

            print(
                f"Dimensions: "
                f"{result['dimensions']}"
            )

            print(
                f"Precision: "
                f"{result['precision']:.0%} "
                f"({result['hits']}/{result['total']})"
            )

            print(
                f"Chunk embedding latency: "
                f"{result['chunk_latency']:.3f}s"
            )

            print(
                f"Query embedding latency: "
                f"{result['query_latency']:.3f}s"
            )

        except Exception as error:

            print(
                f"Model failed: {error}"
            )

    print(
        "\n" + "=" * 80
    )

    print("SUMMARY")

    print("=" * 80)

    print(
        f"{'Model':<30}"
        f"{'Dims':<10}"
        f"{'Precision':<12}"
        f"{'Latency':<12}"
    )

    print("-" * 64)

    for result in all_results:

        total_latency = (
            result["chunk_latency"]
            +
            result["query_latency"]
        )

        print(
            f"{result['model']:<30}"
            f"{result['dimensions']:<10}"
            f"{result['precision']:<12.0%}"
            f"{total_latency:<12.3f}"
        )

    print(
        "\n" + "=" * 80
    )

    print("FAILURE ANALYSIS")

    print("=" * 80)

    for result in all_results:

        print(
            f"\nMODEL: "
            f"{result['model']}"
        )

        failures = [
            item
            for item in result["results"]
            if not item["correct"]
        ]

        if not failures:

            print(
                "No retrieval failures."
            )

            continue

        for failure in failures:

            print(
                "\nQuery:"
            )

            print(
                failure["query"]
            )

            print(
                f"Type: "
                f"{failure['type']}"
            )

            print(
                f"Expected: "
                f"{failure['expected']}"
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