from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("datasets/acme_q3_strategy_memo.txt")

CHUNK_SIZE = 400

OVERLAP_PERCENTAGES = [
    0,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
]


def load_document() -> str:
    return DATA_PATH.read_text(
        encoding="utf-8"
    ).strip()


def create_chunks(
    text: str,
    overlap_percent: int
) -> list[str]:
    """
    Create recursive chunks using a percentage-based overlap.
    """

    overlap_chars = int(
        CHUNK_SIZE * overlap_percent / 100
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=overlap_chars,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    return splitter.split_text(text)


def calculate_total_characters(
    chunks: list[str]
) -> int:
    """
    Count all characters stored across chunks,
    including duplicated overlap.
    """

    return sum(
        len(chunk)
        for chunk in chunks
    )


def calculate_duplication_ratio(
    original_length: int,
    stored_length: int
) -> float:
    """
    Estimate extra text introduced by chunk overlap.
    """

    if original_length == 0:
        return 0.0

    extra = stored_length - original_length

    return max(
        extra / original_length,
        0.0
    )


def show_boundary_example(
    chunks: list[str]
):
    """
    Print one pair of neighboring chunks
    so overlap can be inspected visually.
    """

    if len(chunks) < 2:
        return

    first = chunks[0]
    second = chunks[1]

    print("\nBoundary example:")

    print("\nEnd of Chunk 0:")
    print(first[-150:])

    print("\nStart of Chunk 1:")
    print(second[:150])


def main():

    print("=" * 80)
    print("M04 - CHUNK OVERLAP EXPERIMENT")
    print("=" * 80)

    document = load_document()

    original_length = len(document)

    print(
        f"Original document characters: "
        f"{original_length}"
    )

    print()

    print(
        f"{'Overlap':<12}"
        f"{'Chunks':<12}"
        f"{'Stored Chars':<16}"
        f"{'Extra Text':<14}"
    )

    print("-" * 54)

    for percentage in OVERLAP_PERCENTAGES:

        chunks = create_chunks(
            document,
            percentage
        )

        stored_length = (
            calculate_total_characters(
                chunks
            )
        )

        duplication_ratio = (
            calculate_duplication_ratio(
                original_length,
                stored_length
            )
        )

        print(
            f"{str(percentage) + '%':<12}"
            f"{len(chunks):<12}"
            f"{stored_length:<16}"
            f"{duplication_ratio:.1%}"
        )

        if percentage in [
            0,
            20,
            50
        ]:
            print(
                "\n" + "-" * 54
            )

            print(
                f"{percentage}% OVERLAP"
            )

            show_boundary_example(
                chunks
            )

            print(
                "\n" + "-" * 54
            )


if __name__ == "__main__":
    main()