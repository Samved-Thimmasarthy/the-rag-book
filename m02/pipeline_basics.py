import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_PATH = Path("datasets/acme_q3_strategy_memo.txt")

EMBEDDING_MODEL = "text-embedding-3-small"


# Load secrets from .env
load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

anthropic_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ------------------------------------------------------------
# STAGE 1: INGESTION
# ------------------------------------------------------------

def ingest_document(path: Path) -> str:
    """
    Load a raw text document from disk.
    """

    text = path.read_text(encoding="utf-8")

    return text.strip()


# ------------------------------------------------------------
# STAGE 2: CHUNKING
# ------------------------------------------------------------

def chunk_document(
    text: str,
    chunk_size: int = 120
) -> list[str]:
    """
    Split the document into fixed-size word chunks.
    """

    words = text.split()

    chunks = []

    for start in range(
        0,
        len(words),
        chunk_size
    ):
        chunk_words = words[
            start : start + chunk_size
        ]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks


# ------------------------------------------------------------
# STAGE 3: EMBEDDING
# ------------------------------------------------------------

def embed_texts(
    texts: list[str]
) -> np.ndarray:
    """
    Convert text into numerical embedding vectors.
    """

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    return np.array(embeddings)

# ------------------------------------------------------------
# STAGE 4: RETRIEVAL
# ------------------------------------------------------------

def cosine_similarity(
    query_vector: np.ndarray,
    document_vectors: np.ndarray
) -> np.ndarray:
    """
    Compare one query vector against all document vectors.
    """

    dot_products = document_vectors @ query_vector

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    query_norm = np.linalg.norm(
        query_vector
    )

    similarities = dot_products / (
        document_norms * query_norm
    )

    return similarities


def retrieve_top_k(
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    k: int = 2
) -> list[tuple[int, float, str]]:
    """
    Embed the user's query and return
    the k most semantically similar chunks.
    """

    query_embedding = embed_texts(
        [query]
    )[0]

    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    top_indices = ranked_indices[:k]

    results = []

    for index in top_indices:
        results.append(
            (
                int(index),
                float(similarities[index]),
                chunks[index]
            )
        )

    return results

# ------------------------------------------------------------
# STAGE 5: GENERATION
# ------------------------------------------------------------

def generate_answer(
    query: str,
    retrieved_results: list[tuple[int, float, str]]
) -> str:
    """
    Generate a grounded answer using only retrieved context.
    """

    context_parts = []

    for chunk_index, score, text in retrieved_results:
        context_parts.append(
            f"[Chunk {chunk_index} | Similarity {score:.4f}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are answering questions about Acme Corporation.

Use only the context provided below.

If the context does not contain enough information,
say that the answer cannot be determined from the provided context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("M02 - BASIC RAG PIPELINE")
    print("=" * 70)

    # Stage 1
    document = ingest_document(DATA_PATH)

    print("\n[STAGE 1 - INGESTION]")
    print(
        f"Loaded document with "
        f"{len(document)} characters."
    )

    # Stage 2
    chunks = chunk_document(document)

    print("\n[STAGE 2 - CHUNKING]")
    print(
        f"Created {len(chunks)} chunks."
    )

    for index, chunk in enumerate(chunks):
        print(
            f"Chunk {index}: "
            f"{len(chunk.split())} words"
        )

    # Stage 3
    print("\n[STAGE 3 - EMBEDDING & STORE]")

    chunk_embeddings = embed_texts(chunks)

    print(
        "Embedding matrix shape:",
        chunk_embeddings.shape
    )

    print(
        "Number of stored vectors:",
        len(chunk_embeddings)
    )

    print(
        "Dimensions per vector:",
        chunk_embeddings.shape[1]
    )

    print(
        "\nFirst 10 values of Chunk 0 embedding:"
    )

    print(
        chunk_embeddings[0][:10]
    )
    
    # Stage 4
    print("\n[STAGE 4 - RETRIEVAL]")

    query = "What risks could threaten Acme's Q3 plan?"

    print(f"Query: {query}")

    results = retrieve_top_k(
        query,
        chunks,
        chunk_embeddings,
        k=2
    )

    for rank, (
        chunk_index,
        score,
        text
    ) in enumerate(results, start=1):

        print(
            f"\nResult #{rank}"
        )

        print(
            f"Chunk: {chunk_index}"
        )

        print(
            f"Similarity: {score:.4f}"
        )

        print(
            f"Text: {text[:300]}"
        )

    # Stage 5
    print("\n[STAGE 5 - GENERATION]")

    answer = generate_answer(
        query,
        results
    )

    print("\nFinal grounded answer:")
    print(answer)
        


if __name__ == "__main__":
    main()