"""
Stage 7: Answer Generation
--------------------------
Takes the user's question plus the retrieved context chunks and asks
a language model to produce a grounded answer.

Supports three providers, checked in this order:
1. GROQ_API_KEY      -> uses Groq (free tier, e.g. Llama models)
2. GOOGLE_API_KEY    -> uses Gemini (google-generativeai)
3. ANTHROPIC_API_KEY -> uses Claude (anthropic)

All are read from environment variables (loaded from a local .env file,
which is NOT committed/shared) -- never hardcoded in source.

If none are set, this module falls back to a simple extractive answer
built directly from the retrieved chunks, so the pipeline still runs
end-to-end for demo purposes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same folder as this script, regardless of where you
# run `python main.py` from. This avoids the common bug where the key
# "isn't found" just because the terminal's current directory is different
# from the project folder.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from the user's documents. If the context does not "
    "contain the answer, say so clearly instead of guessing. Keep answers "
    "concise and cite which part of the context you used when helpful."
)


def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Context {i} | source: {chunk.source} | relevance: {score:.2f}]\n{chunk.text}"
        )
    context_text = "\n\n".join(context_blocks)

    return (
        f"Context from documents:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context above."
    )


def generate_answer(question: str, retrieved_chunks: list,
                     anthropic_model: str = "claude-sonnet-4-6",
                     gemini_model: str = "gemini-2.0-flash",
                     groq_model: str = "llama-3.3-70b-versatile") -> str:
    prompt = build_prompt(question, retrieved_chunks)

    groq_key = os.environ.get("GROQ_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if groq_key:
        return _generate_with_groq(prompt, groq_key, groq_model, retrieved_chunks, question)
    elif google_key:
        return _generate_with_gemini(prompt, google_key, gemini_model, retrieved_chunks, question)
    elif anthropic_key:
        return _generate_with_claude(prompt, anthropic_key, anthropic_model, retrieved_chunks, question)
    else:
        return _fallback_answer(question, retrieved_chunks)


def _generate_with_groq(prompt: str, api_key: str, model: str,
                         retrieved_chunks: list, question: str) -> str:
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq call failed: {e}]\n\n" + _fallback_answer(question, retrieved_chunks)


def _generate_with_gemini(prompt: str, api_key: str, model: str,
                           retrieved_chunks: list, question: str) -> str:
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
        response = gen_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini call failed: {e}]\n\n" + _fallback_answer(question, retrieved_chunks)


def _generate_with_claude(prompt: str, api_key: str, model: str,
                           retrieved_chunks: list, question: str) -> str:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[Claude call failed: {e}]\n\n" + _fallback_answer(question, retrieved_chunks)


def _fallback_answer(question: str, retrieved_chunks: list) -> str:
    """Simple extractive fallback used when no API key is configured."""
    lines = [
        "(No GROQ_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY found — showing raw retrieved context instead of a generated answer.)",
        f"Question: {question}",
        "",
        "Most relevant excerpts found:",
    ]
    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        snippet = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
        lines.append(f"\n{i}. (source: {chunk.source}, score: {score:.2f})\n   {snippet}")
    return "\n".join(lines)
