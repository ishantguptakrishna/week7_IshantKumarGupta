"""
Quick diagnostic: run this to check whether your .env / API key setup
is actually working, without needing to run the full pipeline.

Usage:
    python check_setup.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
print(f"Looking for .env at: {env_path}")
print(f".env exists: {env_path.exists()}\n")

load_dotenv(dotenv_path=env_path)

google_key = os.environ.get("GOOGLE_API_KEY")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

if groq_key:
    print(f"GROQ_API_KEY found, starts with: {groq_key[:6]}... (length {len(groq_key)})")
else:
    print("GROQ_API_KEY: NOT FOUND")

if google_key:
    print(f"GOOGLE_API_KEY found, starts with: {google_key[:6]}... (length {len(google_key)})")
else:
    print("GOOGLE_API_KEY: NOT FOUND")

if anthropic_key:
    print(f"ANTHROPIC_API_KEY found, starts with: {anthropic_key[:6]}... (length {len(anthropic_key)})")
else:
    print("ANTHROPIC_API_KEY: NOT FOUND")

if not google_key and not anthropic_key and not groq_key:
    print(
        "\nNo key detected. Make sure:\n"
        "  1. The file is named exactly '.env' (not '.env.txt')\n"
        "  2. It's in the same folder as this script\n"
        "  3. It contains a line like: GOOGLE_API_KEY=your-key-here\n"
        "     (no quotes, no spaces around the =)"
    )
else:
    print("\nKey detected successfully. If main.py still isn't generating real "
          "answers, the key itself may be invalid or missing permissions.")
