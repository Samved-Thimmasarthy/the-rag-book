import json
import os
import re

import anthropic
from dotenv import load_dotenv


CLAUDE_MODEL = "claude-sonnet-4-6"


TEST_QUERIES = [
    "What is Product Alpha's ARR?",
    "How's the company doing?",
    "Why are we losing deals?",
    "Tell me about risks.",
    "Should we delay the September launch given competitive pressure?",
    "What is Beta's churn rate and how does it compare to Alpha?",
    "What should I tell the board?",
    "How many enterprise deals did we close in Q2?",
]


load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
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

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


def classify_query(query: str) -> dict:
    """
    Classify a query and recommend the appropriate
    query-enhancement technique.
    """

    prompt = f"""
Classify the following search query and recommend the best
query-enhancement technique.

Query types:

- factual_precise:
  asks for a specific fact using clear entities or metrics.
  Example: "What is Product Alpha's ARR?"

- factual_vague:
  asks for factual information but uses vague language.
  Example: "How's the company doing?"

- analytical:
  asks why or how something is happening and requires reasoning.
  Example: "Why are we losing deals?"

- exploratory:
  asks broadly about a topic.
  Example: "Tell me about risks."

- complex:
  requires multiple pieces of context, comparison, or
  cross-referencing.
  Example:
  "Should we delay the September launch given competitive pressure?"

Enhancement techniques:

Routing policy:

- factual_precise -> none
- factual_vague -> rewrite
- analytical -> hyde
- exploratory -> rewrite
- complex decision/context question -> step_back
- comparison or clearly separable multi-part question -> multi_query

Important:
Follow this routing policy unless there is a very strong reason not to.
Do not choose multi_query merely because a topic could have multiple causes.
Use HyDE for why/how analytical questions.
Use step_back for decision questions that need broader surrounding context.
Use multi_query mainly for explicit comparisons or clearly separable sub-questions.
Return ONLY valid JSON with exactly these fields:

{{
    "query_type": "...",
    "recommended_technique": "...",
    "confidence": 0.0,
    "reasoning": "..."
}}

Keep reasoning to one short sentence.

Query:
{query}
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=250,
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

        result = json.loads(
            cleaned_text
        )

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
            ),
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
            "reasoning": (
                "Classification failed; "
                "defaulting to rewriting."
            ),
        }


def main():

    print("=" * 80)
    print("M06 - QUERY ROUTER")
    print("=" * 80)

    for query in TEST_QUERIES:

        result = classify_query(
            query
        )

        print(
            "\n" + "-" * 80
        )

        print(
            f"QUERY:\n{query}"
        )

        print(
            f"\nTYPE:\n"
            f"{result['query_type']}"
        )

        print(
            f"\nTECHNIQUE:\n"
            f"{result['recommended_technique']}"
        )

        print(
            f"\nCONFIDENCE:\n"
            f"{result['confidence']:.0%}"
        )

        print(
            f"\nREASONING:\n"
            f"{result['reasoning']}"
        )


if __name__ == "__main__":
    main()