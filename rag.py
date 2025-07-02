"""Minimal RAG example using Fireworks via LangChain."""
import sys
from langchain_fireworks import ChatFireworks


def main(prompt: str) -> None:
    """Query Fireworks with a prompt and print the result."""
    llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-8b-instruct")
    response = llm.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag.py '<prompt>'")
        sys.exit(1)
    main(sys.argv[1])
