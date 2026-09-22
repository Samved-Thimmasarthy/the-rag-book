import os

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
    "Why might we miss our targets?",
    "What should I tell the board?",
    "What is Alpha's ARR?",
]


load_dotenv()

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_hypothetical_answer(query: str) -> str:
    prompt = f"""
Write a short hypothetical answer to the question below as if
you were writing an internal business strategy memo.

Requirements:
- Write 3 to 5 sentences.
- Use concrete business terminology.
- Mention plausible categories of evidence that could answer the question.
- Do not claim this is the real answer.
- The purpose is retrieval, not factual accuracy.

Question:
{query}

Hypothetical answer:
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

    return message.content[0].text.strip()


def embed_texts(texts: list[str]) -> np.ndarray:

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


def merge_results(
    baseline_results: list[tuple[int, float]],
    hyde_results: list[tuple[int, float]],
    top_k: int = 5
) -> list[tuple[int, float]]:

    best_scores = {}

    for chunk_index, score in (
        baseline_results
        +
        hyde_results
    ):

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

    return ranked[:top_k]


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
    print("M06 - HyDE RETRIEVAL")
    print("=" * 80)

    print(
        "\nEmbedding document chunks once..."
    )

    chunk_embeddings = embed_texts(
        CHUNKS
    )

    for query in TEST_QUERIES:

        print(
            "\n" + "=" * 80
        )

        print(
            f"QUERY:\n{query}"
        )

        baseline_embedding = embed_texts(
            [query]
        )[0]

        baseline_results = retrieve(
            baseline_embedding,
            chunk_embeddings
        )

        hypothetical_answer = (
            generate_hypothetical_answer(
                query
            )
        )

        print(
            "\nHYPOTHETICAL ANSWER:"
        )

        print(
            hypothetical_answer
        )

        hyde_embedding = embed_texts(
            [hypothetical_answer]
        )[0]

        hyde_results = retrieve(
            hyde_embedding,
            chunk_embeddings
        )

        combined_results = merge_results(
            baseline_results,
            hyde_results
        )

        print_results(
            "BASELINE RETRIEVAL",
            baseline_results
        )

        print_results(
            "HyDE RETRIEVAL",
            hyde_results
        )

        print_results(
            "COMBINED BASELINE + HyDE",
            combined_results
        )


if __name__ == "__main__":
    main()