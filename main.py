from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.llm_openrouter import OpenRouterLLM

SAMPLE_FILES = [
    # "data/python_intro.txt",
    # "data/vector_store_notes.md",
    # "data/rag_system_design.md",
    # "data/customer_support_playbook.txt",
    # "data/chunking_experiment_report.md",
    # "data/vi_retrieval_notes.md",
    "data/Chính sách bảo vệ dữ liệu cá nhân.txt",
    "data/Dành cho tài xế bike.txt",
    "data/ĐIỀU KHOẢN CHUNG HỢP ĐỒNG DỊCH VỤ.txt",
    "data/ĐIỀU KHOẢN CHUNG.txt",
    "data/donhang_promt.txt",
    "data/khach_hang.txt",
    "data/nhahang.txt",
    "data/Quy tắc bảo vệ quyền lợi người tiêu dùng.txt",
    "data/taxi.txt",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    # agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    llm = OpenRouterLLM(model=model)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


# def main() -> int:
#     question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
#     return run_manual_demo(question=question)

def main() -> int:
    # Nếu có query truyền từ CLI → chạy 1 lần
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        return run_manual_demo(question=question)

    # Interactive mode
    print("\n=== Interactive RAG Mode ===")
    print("Type your question and press Enter")
    print("Type 'exit' or 'quit' to stop\n")

    # Setup 1 lần duy nhất (IMPORTANT: tránh rebuild store mỗi lần)
    load_dotenv(override=False)

    docs = load_documents_from_files(SAMPLE_FILES)
    if not docs:
        print("No documents loaded.")
        return 1

    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    llm = OpenRouterLLM(model=model)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)

    print(f"\nLoaded {len(docs)} docs | Store size: {store.get_collection_size()}\n")

    # 🔥 LOOP CHÍNH
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("\n🔎 Retrieval:")
        results = store.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"{i}. score={r['score']:.3f} | {r['metadata'].get('source')}")

        print("\n🤖 Answer:")
        answer = agent.answer(query, top_k=3)
        print(answer)
        print("\n" + "-" * 50 + "\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
