import json
import os
import re
import time

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
    "What is Product Alpha's ARR?",
    "How's the company doing?",
    "Why are we losing deals?",
    "Tell me about risks.",
    "Should we delay the September launch given competitive pressure?",
    "What is Beta's churn rate and how does it compare to Alpha?",
]


load_dotenv()

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def clean_json_response(
    text: str,
    opening: str,
    closing: str
) -> str:
    """
    Remove markdown fences and extract the JSON portion
    of a Claude response.
    """

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

    start = text.find(opening)
    end = text.rfind(closing)

    if start != -1 and end != -1:
        return text[start:end + 1]

    return text


def classify_query(query: str) -> dict:
    """
    Classify the query and choose an enhancement strategy.
    """

    prompt = f"""
Classify the search query and recommend an enhancement technique.

Query types:
- factual_precise
- factual_vague
- analytical
- exploratory
- complex

Routing policy:

- factual_precise -> none
- factual_vague -> rewrite
- analytical -> hyde
- exploratory -> rewrite
- complex decision/context question -> step_back
- comparison or clearly separable multi-part question -> multi_query

Important:
Do not choose multi_query merely because there could be multiple causes.
Use HyDE for why/how analytical questions.
Use step_back for decision questions that require broader context.
Use multi_query mainly for explicit comparisons or separable sub-questions.

Return ONLY JSON:

{{
    "query_type": "...",
    "recommended_technique": "...",
    "confidence": 0.0,
    "reasoning": "..."
}}

Query:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=250,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = message.content[0].text.strip()

    cleaned = clean_json_response(
        raw,
        "{",
        "}"
    )

    try:
        result = json.loads(cleaned)

        return {
            "query_type": result.get(
                "query_type",
                "unknown"
            ),
            "recommended_technique": result.get(
                "recommended_technique",
                "rewrite"
            ),
            "confidence": float(
                result.get(
                    "confidence",
                    0.5
                )
            ),
            "reasoning": result.get(
                "reasoning",
                "No reasoning provided."
            )
        }

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):
        return {
            "query_type": "unknown",
            "recommended_technique": "rewrite",
            "confidence": 0.5,
            "reasoning": "Classification failed; defaulting to rewrite."
        }


def embed_texts(
    texts: list[str]
) -> np.ndarray:
    """
    Embed one or more strings with OpenAI.
    """

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
    """
    Compute cosine similarity between two vectors.
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


def retrieve(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    top_k: int = 3
) -> list[tuple[int, float]]:
    """
    Rank document chunks by cosine similarity.
    """

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


def raw_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> list[tuple[int, float]]:
    """
    Standard retrieval using the original query.
    """

    query_embedding = embed_texts(
        [query]
    )[0]

    return retrieve(
        query_embedding,
        chunk_embeddings
    )


def rewrite_query(
    query: str
) -> list[str]:
    """
    Generate three more precise search queries.
    """

    prompt = f"""
Rewrite this query into exactly 3 precise search queries
for an internal business strategy memo.

Rules:
- preserve the original intent
- use concrete business terminology
- do not invent facts
- return ONLY a JSON array

Query:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=250,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = message.content[0].text.strip()

    cleaned = clean_json_response(
        raw,
        "[",
        "]"
    )

    try:
        result = json.loads(cleaned)

        if isinstance(
            result,
            list
        ):
            return [
                str(item).strip()
                for item in result[:3]
                if str(item).strip()
            ]

    except json.JSONDecodeError:
        pass

    return [query]


def rewrite_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> list[tuple[int, float]]:
    """
    Search with both the raw query and rewritten variants.
    """

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

    for embedding in embeddings:
        for chunk_index, score in retrieve(
            embedding,
            chunk_embeddings
        ):
            if (
                chunk_index not in best_scores
                or
                score > best_scores[chunk_index]
            ):
                best_scores[
                    chunk_index
                ] = score

    return sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]


def generate_hypothesis(
    query: str
) -> str:
    """
    Generate a hypothetical document-style answer for HyDE.
    """

    prompt = f"""
Write a short hypothetical business strategy memo answer
to the question below.

The goal is retrieval, not factual correctness.

Use document-style business terminology.
Do not pretend the hypothetical answer is verified.

Question:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        message.content[0]
        .text
        .strip()
    )


def hyde_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> list[tuple[int, float]]:
    """
    Combine raw-query retrieval with HyDE retrieval.
    """

    baseline = raw_retrieval(
        query,
        chunk_embeddings
    )

    hypothesis = generate_hypothesis(
        query
    )

    hypothesis_embedding = embed_texts(
        [hypothesis]
    )[0]

    hyde_results = retrieve(
        hypothesis_embedding,
        chunk_embeddings
    )

    best_scores = {}

    for chunk_index, score in (
        baseline
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

    return sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]


def generate_step_back(
    query: str
) -> str:
    """
    Generate a broader but focused context question.
    """

    prompt = f"""
Generate ONE broader but still focused step-back question
that would help retrieve context needed to answer the
original query.

Rules:
- preserve important entities
- preserve the main topic
- focus on relevant context, status, risks, timeline, or dependencies
- do not make the question overly generic
- return ONLY the question

Original query:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        message.content[0]
        .text
        .strip()
    )


def step_back_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> list[tuple[int, float]]:
    """
    Combine retrieval from the original question and
    the generated step-back question.
    """

    original_results = raw_retrieval(
        query,
        chunk_embeddings
    )

    step_back_question = generate_step_back(
        query
    )

    step_embedding = embed_texts(
        [step_back_question]
    )[0]

    step_results = retrieve(
        step_embedding,
        chunk_embeddings
    )

    best_scores = {}

    for chunk_index, score in (
        original_results
        +
        step_results
    ):
        if (
            chunk_index not in best_scores
            or
            score > best_scores[chunk_index]
        ):
            best_scores[
                chunk_index
            ] = score

    return sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]


def generate_subqueries(
    query: str
) -> list[str]:
    """
    Break a comparison or multi-part query into
    independent searches.
    """

    prompt = f"""
Break the following query into 2 or 3 independent search queries.

Rules:
- preserve the original intent
- separate distinct entities or sub-questions
- do not invent facts
- return ONLY a JSON array

Original query:
{query}
"""

    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = message.content[0].text.strip()

    cleaned = clean_json_response(
        raw,
        "[",
        "]"
    )

    try:
        result = json.loads(
            cleaned
        )

        if isinstance(
            result,
            list
        ):
            return [
                str(item).strip()
                for item in result[:3]
                if str(item).strip()
            ]

    except json.JSONDecodeError:
        pass

    return [query]


def multi_query_retrieval(
    query: str,
    chunk_embeddings: np.ndarray
) -> list[tuple[int, float]]:
    """
    Search independently with generated subqueries
    and merge the best results.
    """

    subqueries = generate_subqueries(
        query
    )

    embeddings = embed_texts(
        subqueries
    )

    best_scores = {}

    for embedding in embeddings:
        for chunk_index, score in retrieve(
            embedding,
            chunk_embeddings
        ):
            if (
                chunk_index not in best_scores
                or
                score > best_scores[chunk_index]
            ):
                best_scores[
                    chunk_index
                ] = score

    return sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]


def execute_strategy(
    query: str,
    technique: str,
    chunk_embeddings: np.ndarray
) -> tuple[
    list[tuple[int, float]],
    str,
    float
]:
    """
    Execute the selected enhancement technique.

    If enhancement fails, safely fall back to raw retrieval.

    Returns:
        results
        technique actually used
        elapsed time
    """

    start = time.perf_counter()

    try:
        if technique == "none":
            results = raw_retrieval(
                query,
                chunk_embeddings
            )

        elif technique == "rewrite":
            results = rewrite_retrieval(
                query,
                chunk_embeddings
            )

        elif technique == "hyde":
            results = hyde_retrieval(
                query,
                chunk_embeddings
            )

        elif technique == "step_back":
            results = step_back_retrieval(
                query,
                chunk_embeddings
            )

        elif technique == "multi_query":
            results = multi_query_retrieval(
                query,
                chunk_embeddings
            )

        else:
            technique = "none"

            results = raw_retrieval(
                query,
                chunk_embeddings
            )

    except Exception as error:
        print(
            f"\nEnhancement failed: "
            f"{type(error).__name__}: {error}"
        )

        print(
            "Falling back to raw retrieval..."
        )

        technique = "fallback_raw"

        results = raw_retrieval(
            query,
            chunk_embeddings
        )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    return (
        results,
        technique,
        elapsed
    )


def print_results(
    results: list[tuple[int, float]]
):
    """
    Print ranked retrieval results.
    """

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
    print("M06 - AUTOMATIC QUERY ENHANCEMENT SELECTOR")
    print("=" * 80)

    print(
        "\nEmbedding document chunks..."
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

        classification = classify_query(
            query
        )

        selected_technique = classification.get(
            "recommended_technique",
            "none"
        )

        print(
            f"\nTYPE:\n"
            f"{classification.get('query_type', 'unknown')}"
        )

        print(
            f"\nSELECTED TECHNIQUE:\n"
            f"{selected_technique}"
        )

        print(
            f"\nCONFIDENCE:\n"
            f"{classification.get('confidence', 0):.0%}"
        )

        print(
            f"\nROUTER REASONING:\n"
            f"{classification.get('reasoning', 'N/A')}"
        )

        (
            results,
            technique_used,
            elapsed
        ) = execute_strategy(
            query,
            selected_technique,
            chunk_embeddings
        )

        print(
            f"\nACTUAL TECHNIQUE USED:\n"
            f"{technique_used}"
        )

        print(
            f"\nENHANCEMENT + RETRIEVAL LATENCY:\n"
            f"{elapsed:.3f}s"
        )

        print(
            "\nRETRIEVAL RESULTS:"
        )

        print_results(
            results
        )


if __name__ == "__main__":
    main()