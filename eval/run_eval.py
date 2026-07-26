"""
Runs the eval question set through the RAG chain and scores each answer by
keyword overlap against expected_keywords. This is a crude but useful
regression check — run it after any change to chunking, retrieval weights,
or the reranker to catch quality drops before they reach the app.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --output eval/results.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.retrieval import load_vector_db, build_reranked_hybrid_retriever
from src.chain import build_llm, build_rag_chain

EVAL_SET_PATH = Path(__file__).parent / "eval_questions.json"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate(rag_chain, eval_set: list[dict]) -> pd.DataFrame:
    results = []
    for item in eval_set:
        result = rag_chain.invoke({"input": item["question"], "chat_history": []})
        answer = result["answer"]
        answer_lower = answer.lower()

        hits = [kw for kw in item["expected_keywords"] if kw.lower() in answer_lower]
        score = len(hits) / len(item["expected_keywords"]) if item["expected_keywords"] else 0.0

        results.append(
            {
                "question": item["question"],
                "answer": answer,
                "expected_keywords": item["expected_keywords"],
                "keywords_found": hits,
                "score": score,
            }
        )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG chain against the eval set.")
    parser.add_argument("--groq-api-key", type=str, default=None, help="Groq API key (falls back to GROQ_API_KEY env var)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as CSV")
    args = parser.parse_args()

    import os
    groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Provide --groq-api-key or set the GROQ_API_KEY environment variable.")

    print("Loading pipeline...")
    vector_db = load_vector_db()
    retriever = build_reranked_hybrid_retriever(vector_db)
    llm = build_llm(groq_api_key)
    rag_chain = build_rag_chain(llm, retriever)

    eval_set = load_eval_set()
    print(f"Running {len(eval_set)} eval questions...")
    df = evaluate(rag_chain, eval_set)

    print(f"\nAverage score: {df['score'].mean():.2%}")
    print(df[["question", "score", "keywords_found"]].to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()