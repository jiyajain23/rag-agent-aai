"""
Runs the eval question set through the RAG chain and scores each answer by
keyword overlap against expected_keywords. This is a crude but useful
regression check — run it after any change to chunking, retrieval weights,
or the reranker to catch quality drops before they reach the app.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --output eval/results/my_run.csv
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.retrieval import load_vector_db, build_reranked_hybrid_retriever
from src.chain import build_llm, build_rag_chain

EVAL_SET_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent.parent / "test-results"


def _load_groq_key_from_secrets() -> str | None:
    """Try to read GROQ_API_KEY from .streamlit/secrets.toml (dev convenience)."""
    secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # fallback for older Python
        except ImportError:
            return None
    with open(secrets_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("GROQ_API_KEY")


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate(rag_chain, eval_set: list[dict]) -> pd.DataFrame:
    results = []
    for i, item in enumerate(eval_set, 1):
        print(f"  [{i}/{len(eval_set)}] {item['question'][:70]}...")
        result = rag_chain.invoke({"input": item["question"], "chat_history": []})
        answer = result["answer"]
        answer_lower = answer.lower()

        hits = [kw for kw in item["expected_keywords"] if kw.lower() in answer_lower]
        score = len(hits) / len(item["expected_keywords"]) if item["expected_keywords"] else 0.0

        results.append(
            {
                "question": item["question"],
                "answer": answer,
                "expected_keywords": ", ".join(item["expected_keywords"]),
                "keywords_found": ", ".join(hits),
                "score": score,
            }
        )
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(output_path, index=False)

    # Save human-readable text summary alongside the CSV
    summary_path = output_path.with_suffix(".txt")
    avg_score = df["score"].mean()
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Eval Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions: {len(df)}  |  Average score: {avg_score:.2%}\n")
        f.write("=" * 80 + "\n\n")
        for _, row in df.iterrows():
            f.write(f"Q: {row['question']}\n")
            f.write(f"A: {row['answer'].strip()}\n")
            f.write(f"   Score: {row['score']:.0%}  |  Found: [{row['keywords_found']}]  |  Expected: [{row['expected_keywords']}]\n")
            f.write("-" * 80 + "\n\n")

    print(f"\nResults saved to:\n  CSV:  {output_path}\n  Text: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG chain against the eval set.")
    parser.add_argument("--groq-api-key", type=str, default=None, help="Groq API key (falls back to GROQ_API_KEY env var or .streamlit/secrets.toml)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results CSV (default: eval/results/<timestamp>.csv)")
    args = parser.parse_args()

    # Resolve API key: CLI arg → env var → secrets.toml
    groq_api_key = (
        args.groq_api_key
        or os.environ.get("GROQ_API_KEY")
        or _load_groq_key_from_secrets()
    )
    if not groq_api_key:
        raise ValueError(
            "No Groq API key found. Provide --groq-api-key, set GROQ_API_KEY env var, "
            "or add it to .streamlit/secrets.toml."
        )

    # Resolve output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{timestamp}.csv"

    print("Loading pipeline...")
    vector_db = load_vector_db()
    retriever = build_reranked_hybrid_retriever(vector_db)
    llm = build_llm(groq_api_key)
    rag_chain = build_rag_chain(llm, retriever)

    eval_set = load_eval_set()
    print(f"Running {len(eval_set)} eval questions...\n")
    df = evaluate(rag_chain, eval_set)

    avg = df["score"].mean()
    print(f"\n{'='*50}")
    print(f"Average score: {avg:.2%}")
    print(f"{'='*50}")
    print(df[["question", "score", "keywords_found"]].to_string(index=False))

    save_results(df, output_path)


if __name__ == "__main__":
    main()