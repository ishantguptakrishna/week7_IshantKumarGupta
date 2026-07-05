# Document Question-Answering System (RAG)

A simple, beginner-friendly Retrieval-Augmented Generation pipeline that answers
questions about your own PDFs or text files.

## How it maps to the project stages

| Stage                | File               |
|-----------------------|-------------------|
| 1. Document Ingestion  | `loader.py`        |
| 2. Text Chunking       | `chunker.py`        |
| 3. Embedding Creation  | `vector_store.py`   |
| 4. Vector Database     | `vector_store.py`   |
| 5. Query Processing    | `vector_store.py` (`search`) |
| 6. Context Retrieval   | `vector_store.py` (`search`) |
| 7. Answer Generation   | `generator.py`      |

`rag_pipeline.py` wires all the stages together. `main.py` is the command-line
interface you actually run.

## Setup

```bash
pip install -r requirements.txt
```

(Optional but recommended) To get real generated answers instead of raw
retrieved text, set an API key. Three providers are supported — pick one:

- **Groq (recommended, free, no credit card)**: get a key at
  https://console.groq.com/keys, then set `GROQ_API_KEY` in `.env`.
- **Google Gemini**: get a key at https://aistudio.google.com/apikey
  (use this URL, not console.cloud.google.com, or you may get a key with
  a 0-request free quota), then set `GOOGLE_API_KEY` in `.env`.
- **Claude/Anthropic**: get a key at https://console.anthropic.com,
  then set `ANTHROPIC_API_KEY` in `.env`.

Copy `.env.example` to `.env` and fill in whichever key you're using:

```bash
cp .env.example .env
```

Run `python check_setup.py` any time to confirm your key is being detected correctly.

Without a key, the system still runs end-to-end — it just falls back to
showing you the top retrieved passages instead of a generated answer, which is
still useful for understanding what retrieval is doing.

## Usage

Ask questions about a single document:

```bash
python main.py path/to/your_notes.pdf
```

Or about every document in a folder:

```bash
python main.py path/to/your_documents_folder/
```

Then just type questions:

```
Q: What is the main idea of the document?
A: ...
Q: exit
```

## Using it in your own code

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(chunk_size=500, overlap=100, top_k=3)
pipeline.ingest_file("my_resume.pdf")

answer = pipeline.ask("What projects has this person worked on?")
print(answer)

# or, if you want to see which chunks were used:
result = pipeline.ask_with_sources("What projects has this person worked on?")
print(result["answer"])
for src in result["sources"]:
    print(src["source"], src["score"], src["text"][:100])
```

## Design choices (and how to extend them)

- **Embeddings**: uses scikit-learn's TF-IDF vectorizer instead of a neural
  embedding model. This keeps the project dependency-light and fully offline
  (no model downloads), which is ideal for a first RAG project. TF-IDF is
  keyword-based rather than truly semantic, so it works best when the
  question shares vocabulary with the document.
  - *Upgrade path*: swap `TfidfVectorStore` in `vector_store.py` for one built
    on `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) to get real semantic
    similarity, or plug in a hosted vector DB like Chroma, Pinecone, or Weaviate.
- **Chunking**: simple word-count based chunks with overlap, adjustable via
  `chunk_size` and `overlap`. Try smaller chunks for precise fact lookup,
  larger chunks for questions needing more surrounding context.
- **Generation**: uses Claude via the Anthropic API. Swap `generator.py` for
  any other LLM API (OpenAI, local Ollama model, etc.) by changing the
  `generate_answer` function — the rest of the pipeline doesn't need to change.
- **Retrieval quality experiments** (from the project brief): try hybrid
  search (combine TF-IDF keyword scores with embedding similarity), add a
  re-ranking step over the top-k results, or try different `chunk_size`/`overlap`
  values and compare answer quality.

## Files

```
rag_project/
├── loader.py         # Stage 1: load PDFs/text files
├── chunker.py         # Stage 2: split into overlapping chunks
├── vector_store.py    # Stages 3-4: TF-IDF embeddings + similarity search
├── generator.py        # Stage 7: generate grounded answers with Claude
├── rag_pipeline.py      # Orchestrates all stages
├── main.py              # CLI entry point
├── requirements.txt
└── sample_docs/
    └── sample_notes.txt  # small example doc to try the system on
```
