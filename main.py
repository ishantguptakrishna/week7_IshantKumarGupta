"""
Command-line entry point.

Usage:
    python main.py path/to/document.pdf
    python main.py path/to/folder_of_documents/

Then type questions at the prompt. Type 'exit' to quit.
"""

import sys
import os
from rag_pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-pdf-or-folder>")
        sys.exit(1)

    path = sys.argv[1]
    pipeline = RAGPipeline(chunk_size=150, overlap=30, top_k=3)

    if os.path.isdir(path):
        pipeline.ingest_folder(path)
    else:
        pipeline.ingest_file(path)

    print("\nRAG system ready. Ask a question about your document(s) (type 'exit' to quit).\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = pipeline.ask(question)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
