from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("datasets/acme_q3_strategy_memo.txt")


def load_document() -> str:
    return DATA_PATH.read_text(
        encoding="utf-8"
    ).strip()


def fixed_size_chunks(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> list[str]:
    """
    Split text into fixed-size character chunks with overlap.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def sentence_based_chunks(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> list[str]:
    """
    Keep sentence boundaries intact where possible.
    """

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    return splitter.split_text(
        text
    )


def recursive_chunks(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> list[str]:
    """
    Try natural separators before falling back to smaller units.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    return splitter.split_text(
        text
    )


def show_chunks(
    name: str,
    chunks: list[str]
):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Created {len(chunks)} chunks."
    )

    for index, chunk in enumerate(
        chunks[:5]
    ):
        print(
            f"\nChunk {index}"
        )

        print(
            f"Characters: {len(chunk)}"
        )

        print(chunk)


def main():

    print("=" * 70)
    print("M04 - CHUNKING STRATEGIES")
    print("=" * 70)

    document = load_document()

    print(
        f"Loaded document with "
        f"{len(document)} characters."
    )

    fixed = fixed_size_chunks(
        document
    )

    sentence = sentence_based_chunks(
        document
    )

    recursive = recursive_chunks(
        document
    )

    show_chunks(
        "FIXED-SIZE CHUNKING",
        fixed
    )

    show_chunks(
        "SENTENCE-BASED CHUNKING",
        sentence
    )

    show_chunks(
        "RECURSIVE CHUNKING",
        recursive
    )


if __name__ == "__main__":
    main()