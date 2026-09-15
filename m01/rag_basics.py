import os

from anthropic import Anthropic
from dotenv import load_dotenv


# Load values such as ANTHROPIC_API_KEY from the .env file.
load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


question = "What was Product Alpha's Q2 revenue?"

acme_context = """
Acme Corporation Q3 Strategy Memo

Product Alpha achieved Q2 revenue of $12.4M.
Current ARR is $48.2M.
Enterprise adoption remained strong during Q2.
"""


def ask_without_context(question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
    )

    return response.content[0].text


def ask_with_context(question: str, context: str) -> str:
    prompt = f"""
Answer the question using only the provided context.

If the answer is not contained in the context, say:
"I don't know based on the provided context."

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text


print("=" * 60)
print("WITHOUT CONTEXT")
print("=" * 60)
print(ask_without_context(question))

print()

print("=" * 60)
print("WITH CONTEXT")
print("=" * 60)
print(ask_with_context(question, acme_context))


print()

print("=" * 60)
print("CONTEXT DOES NOT CONTAIN ANSWER")
print("=" * 60)

missing_question = "What was Product Alpha's gross margin?"

print(ask_with_context(missing_question, acme_context))

