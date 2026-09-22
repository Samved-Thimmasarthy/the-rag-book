import json
import os
import re

import anthropic
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


CLAUDE_MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "text-embedding-3-small"


CHUNKS = [
    (
        "Acme Corporation enters Q3 2024 with strong momentum "
        "across the product portfolio. Product Alpha achieved "
        "Q2 revenue of $12.4M and ARR reached $48.2M."
    ),
    (
        "Product Beta generated $3.8M in Q2 revenue. "
        "Customer churn increased to 4.2%, creating a "
        "retention concern."
    ),
    (
        "Product Gamma remains in private beta with 47 design "
        "partners. Launch is scheduled for September 15."
    ),
    (
        "Combined Q3 revenue target is $18.6M. Product Alpha "
        "target is $14.2M, Product Beta target is $4.0M, "
        "and Product Gamma target is $0.4M."
    ),
    (
        "Competitive threat from Nexus Corp's enterprise "
        "offering may affect Product Alpha's market position."
    ),
    (
        "Beta churn exceeding the 5% threshold is a risk. "
        "Mitigation includes dedicated customer success "
        "support for at-risk accounts."
    ),
    (
        "Gamma launch delay beyond September 30 is a risk. "
        "Mitigation includes weekly launch readiness reviews."
    ),
    (
        "Key person dependency on Alpha's machine learning "
        "infrastructure team is an operational risk."
    ),
    (
        "Regulatory changes in the EU could affect data "
        "processing and create compliance risk."
    ),
]


TEST_QUERIES = [
    {
        "query": "What is Alpha's ARR?",
        "expected": "$48.2M",
    },
    {
        "query": "How's the company doing?",
        "expected": "strong momentum",
    },
    {
        "query": "Why might we miss our targets?",
        "expected": "Competitive threat",
    },
    {
        "query": "Tell me about risks.",
        "expected": "risk",
    },
    {
        "query": "What should I tell the board?",
        "expected": "strong momentum",
    },
]


load_dotenv()

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def clean_json_response(text: str) -> str:

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


def rewrite_query(query: str) -> list[str]:

    prompt = f"""
Rewrite the following user question into exactly 3 precise search queries
for retrieving information from an internal business strategy memo.

Rules:
- Preserve the original intent.
- Use concrete business terminology.
- Do not invent facts.
- Make each rewrite useful for semantic search.
- Return ONLY a JSON array of exactly 3 strings.
- Do not include markdown fences.
- Do not include explanations.

User question:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    raw_text = (
        message.content[0]
        .text
        .strip()
    )

    cleaned_text = clean_json_response(
        raw_text
    )

    try:

        rewrites = json.loads(
            cleaned_text
        )

        if not isinstance(
            rewrites,
            list
        ):
            return [query]

        rewrites = [
            str(item).strip()
            for item in rewrites
            if str(item).strip()
        ]

        return rewrites[:3] if rewrites else [query]

    except json.JSONDecodeError:
        return [query]


def embed_texts(
    texts: list[str]
) -> np.ndarray:

    response = openai_client.embeddings.create(
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

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


def retrieve(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    top_k: int = 3
) -> list[tuple[int, float]]:

    scores = []

    for index, chunk_embedding in enumerate(
        chunk_embeddings
    ):

        score = cosine_similarity(
            query_embedding,
            chunk_embedding
        )

        scores.append(
            (
                index,
                score
            )
        )

    scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return scores[:top_k]


def baseline_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> list[tuple[int, float]]:

    query_embedding = embed_texts(
        [query]
    )[0]

    return retrieve(
        query_embedding,
        chunk_embeddings
    )


def rewritten_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> tuple[
    list[str],
    list[tuple[int, float]]
]:

    rewrites = rewrite_query(
        query
    )

    all_queries = [
        query
    ] + rewrites

    embeddings = embed_texts(
        all_queries
    )

    best_scores = {}

    for query_embedding in embeddings:

        results = retrieve(
            query_embedding,
            chunk_embeddings
        )

        for chunk_index, score in results:

            if (
                chunk_index not in best_scores
                or
                score > best_scores[chunk_index]
            ):
                best_scores[
                    chunk_index
                ] = score

    ranked = sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return (
        rewrites,
        ranked[:3]
    )


def contains_expected(
    results: list[tuple[int, float]],
    expected: str
) -> bool:

    for chunk_index, _ in results:

        if expected.lower() in CHUNKS[
            chunk_index
        ].lower():
            return True

    return False


def print_results(
    label: str,
    results: list[tuple[int, float]]
):

    print(
        f"\n{label}"
    )

    for rank, (
        chunk_index,
        score
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            CHUNKS[chunk_index]
        )


def main():

    print("=" * 80)
    print("M06 - RAW QUERY VS QUERY REWRITING")
    print("=" * 80)

    print(
        "\nEmbedding document chunks once..."
    )

    chunk_embeddings = embed_texts(
        CHUNKS
    )

    baseline_hits = 0
    rewritten_hits = 0

    for test in TEST_QUERIES:

        query = test["query"]
        expected = test["expected"]

        print(
            "\n" + "=" * 80
        )

        print(
            f"QUERY:\n{query}"
        )

        print(
            f"\nExpected keyword:\n"
            f"{expected}"
        )

        baseline_results = baseline_retrieval(
            query,
            chunk_embeddings
        )

        (
            rewrites,
            rewritten_results
        ) = rewritten_retrieval(
            query,
            chunk_embeddings
        )

        print(
            "\nGenerated rewrites:"
        )

        for index, rewrite in enumerate(
            rewrites,
            start=1
        ):
            print(
                f"{index}. {rewrite}"
            )

        print_results(
            "BASELINE RETRIEVAL",
            baseline_results
        )

        print_results(
            "REWRITTEN RETRIEVAL",
            rewritten_results
        )

        baseline_correct = contains_expected(
            baseline_results,
            expected
        )

        rewritten_correct = contains_expected(
            rewritten_results,
            expected
        )

        if baseline_correct:
            baseline_hits += 1

        if rewritten_correct:
            rewritten_hits += 1

        print(
            "\nRESULT:"
        )

        print(
            f"Baseline hit: "
            f"{baseline_correct}"
        )

        print(
            f"Rewrite hit: "
            f"{rewritten_correct}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    total = len(
        TEST_QUERIES
    )

    print(
        f"Baseline retrieval: "
        f"{baseline_hits}/{total}"
    )

    print(
        f"Rewritten retrieval: "
        f"{rewritten_hits}/{total}"
    )


if __name__ == "__main__":
    main()