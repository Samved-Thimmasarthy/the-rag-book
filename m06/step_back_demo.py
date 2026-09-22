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
    "Should we delay the September launch?",
    "Should Product Gamma launch as planned?",
    "What risks could affect Gamma's launch?",
]


load_dotenv()

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_step_back_question(
    query: str
) -> str:
    """
    Generate a broader but still focused question that helps
    retrieve background context needed to answer the original.
    """

    prompt = f"""
Given the user's question below, generate ONE broader
step-back question that would help retrieve the background
context needed to answer it.

Rules:
- Broaden the question, but keep it focused.
- Do not make it generic.
- Preserve the important entity or topic.
- Focus on status, context, risks, timeline, or dependencies.
- Return ONLY the step-back question.

Original question:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return (
        message.content[0]
        .text
        .strip()
    )


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
    top_k: int = 4
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
    original_results: list[tuple[int, float]],
    step_back_results: list[tuple[int, float]],
    top_k: int = 5
) -> list[tuple[int, float]]:

    best_scores = {}

    for chunk_index, score in (
        original_results
        +
        step_back_results
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


def generate_final_answer(
    original_query: str,
    results: list[tuple[int, float]]
) -> str:

    context = "\n\n".join(
        [
            CHUNKS[chunk_index]
            for chunk_index, _ in results
        ]
    )

    prompt = f"""
Answer the original question using ONLY the context below.

If the context does not support a definite yes/no answer,
say that clearly and explain what the evidence does support.

Original question:
{original_query}

Context:
{context}

Answer:
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return (
        message.content[0]
        .text
        .strip()
    )


def main():

    print("=" * 80)
    print("M06 - STEP-BACK PROMPTING")
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
            f"ORIGINAL QUERY:\n{query}"
        )

        original_embedding = embed_texts(
            [query]
        )[0]

        original_results = retrieve(
            original_embedding,
            chunk_embeddings
        )

        step_back_question = (
            generate_step_back_question(
                query
            )
        )

        print(
            "\nSTEP-BACK QUESTION:"
        )

        print(
            step_back_question
        )

        step_back_embedding = embed_texts(
            [step_back_question]
        )[0]

        step_back_results = retrieve(
            step_back_embedding,
            chunk_embeddings
        )

        combined_results = merge_results(
            original_results,
            step_back_results
        )

        print_results(
            "ORIGINAL QUERY RETRIEVAL",
            original_results
        )

        print_results(
            "STEP-BACK RETRIEVAL",
            step_back_results
        )

        print_results(
            "COMBINED RETRIEVAL",
            combined_results
        )

        final_answer = generate_final_answer(
            query,
            combined_results
        )

        print(
            "\nFINAL ANSWER:"
        )

        print(
            final_answer
        )


if __name__ == "__main__":
    main()