from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        # pass
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # raise NotImplementedError("Implement KnowledgeBaseAgent.answer")
        results = self.store.search(question, top_k=top_k)

        context = "\n\n".join(
            f"[{i+1}] {r['content']}"
            for i, r in enumerate(results)
        )

        prompt = f"""You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer concisely and base only on the context.
"""

        return self.llm_fn(prompt)