import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

INDEX_NAME = "acme-rag-demo"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MODEL = "text-embedding-3-small"
DATA_PATH = Path("datasets/acme_q3_strategy_memo.txt")

# Load secrets from .env
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(
    api_key=pinecone_api_key
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)



# ------------------------------------------------------------
# INITIALIZE PINECONE
# ------------------------------------------------------------



def create_index_if_needed():
    """
    Create a Pinecone index if it does not already exist.
    """

    existing_indexes = [
        index.name
        for index in pc.list_indexes()
    ]

    if INDEX_NAME not in existing_indexes:

        print(
            f"Creating index: {INDEX_NAME}"
        )

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        print("Index creation requested.")

    else:

        print(
            f"Index already exists: {INDEX_NAME}"
        )


def describe_index():
    """
    Display basic index configuration.
    """

    description = pc.describe_index(
        INDEX_NAME
    )

    print("\n[Index Description]")
    print(description)

# ------------------------------------------------------------
# DOCUMENT PREPARATION
# ------------------------------------------------------------

def ingest_document(path: Path) -> str:
    """
    Read the Acme strategy memo from disk.
    """

    return path.read_text(
        encoding="utf-8"
    ).strip()


def chunk_document(
    text: str,
    chunk_size: int = 120
) -> list[str]:
    """
    Split document into fixed-size word chunks.
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

        chunks.append(
            " ".join(chunk_words)
        )

    return chunks


def embed_texts(
    texts: list[str]
) -> list[list[float]]:
    """
    Convert text chunks into OpenAI embedding vectors.
    """

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]

# ------------------------------------------------------------
# UPSERT VECTORS
# ------------------------------------------------------------

def upsert_chunks(
    chunks: list[str]
):
    """
    Embed document chunks and store them in Pinecone.
    """

    print(
        f"\nEmbedding {len(chunks)} chunks..."
    )

    vectors = embed_texts(chunks)

    records = []

    for index, (
        chunk,
        vector
    ) in enumerate(
        zip(chunks, vectors)
    ):

        records.append(
            {
                "id": f"acme-chunk-{index}",
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "source": "Q3 Strategy Memo",
                    "chunk_index": index,
                    "embedding_model": EMBEDDING_MODEL,
                    "indexed_at": time.time(),
                },
            }
        )

    index = pc.Index(
        INDEX_NAME
    )

    index.upsert(
        vectors=records
    )

    print(
        f"Upserted {len(records)} vectors."
    )

    return index

def show_index_stats(index):
    """
    Display the number of vectors stored.
    """

    time.sleep(2)

    stats = (
        index.describe_index_stats()
    )

    print(
        "\n[Index Stats]"
    )

    print(stats)

def query_index(
    index,
    query: str,
    top_k: int = 2,
    metadata_filter: dict | None = None
):
    """
    Embed a user query and retrieve the most similar chunks from Pinecone.
    Optionally restrict retrieval using metadata filters.
    """

    print("\n[QUERY]")
    print(f"Question: {query}")

    if metadata_filter:
        print(f"Filter: {metadata_filter}")
    else:
        print("Filter: None")

    query_vector = embed_texts(
        [query]
    )[0]

    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }

    if metadata_filter:
        query_params["filter"] = metadata_filter

    results = index.query(
        **query_params
    )

    print("\n[TOP MATCHES]")

    if not results.matches:
        print("No matching vectors found.")
        return results

    for rank, match in enumerate(
        results.matches,
        start=1
    ):

        print(f"\nResult #{rank}")
        print(f"ID: {match.id}")
        print(f"Similarity: {match.score:.4f}")

        print(
            f"Chunk index: "
            f"{int(match.metadata['chunk_index'])}"
        )

        print(
            f"Source: "
            f"{match.metadata['source']}"
        )

        print(
            f"Text: "
            f"{match.metadata['text'][:300]}"
        )

    return results

def main():

    print("=" * 70)
    print("M03 - VECTOR DATABASE BASICS")
    print("=" * 70)

    create_index_if_needed()

    describe_index()

    document = ingest_document(
        DATA_PATH
    )

    chunks = chunk_document(
        document
    )

    print(
        f"\nPrepared {len(chunks)} chunks."
    )

    index = upsert_chunks(
	chunks
    )

    show_index_stats(
        index
    )

    query = (
        "What risks could threaten Acme's Q3 plan?"
    )
   
    print("\n" + "=" * 70)
    print("UNFILTERED SEARCH")
    print("=" * 70)
 
    query_index(
        index,
        query,
        top_k=2
    )
    
    print("\n" + "=" * 70)
    print("FILTERED SEARCH - ONLY CHUNK 3")
    print("=" * 70)

    query_index(
    index,
    query,
    top_k=2,
    metadata_filter={
        "chunk_index": {
            "$eq": 3
        }
    }
)

if __name__ == "__main__":
    main()