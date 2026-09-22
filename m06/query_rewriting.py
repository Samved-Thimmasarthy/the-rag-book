import json
import os
import re

import anthropic
from dotenv import load_dotenv


MODEL = "claude-sonnet-4-6"

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


TEST_QUERIES = [
    "What is Alpha's ARR?",
    "How's the company doing?",
    "Why might we miss our targets?",
    "Tell me about risks.",
    "What should I tell the board?",
]


def clean_json_response(
    text: str
) -> str:
    """
    Remove markdown fences or surrounding text
    so the JSON array can be parsed safely.
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

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[
            start:end + 1
        ]

    return text.strip()


def rewrite_query(
    query: str
) -> list[str]:

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

    message = client.messages.create(
        model=MODEL,
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

    print(
        "\nRAW CLAUDE RESPONSE:"
    )

    print(
        raw_text
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

        if not rewrites:
            return [query]

        return rewrites[:3]

    except json.JSONDecodeError:

        print(
            "\nJSON parsing failed."
        )

        return [query]


def main():

    print("=" * 75)
    print("M06 - QUERY REWRITING")
    print("=" * 75)

    for query in TEST_QUERIES:

        print(
            f"\nOriginal query:\n"
            f"{query}"
        )

        rewrites = rewrite_query(
            query
        )

        print(
            "\nRewritten searches:"
        )

        for index, rewrite in enumerate(
            rewrites,
            start=1
        ):

            print(
                f"{index}. {rewrite}"
            )

        print(
            "\n" + "-" * 75
        )


if __name__ == "__main__":
    main()