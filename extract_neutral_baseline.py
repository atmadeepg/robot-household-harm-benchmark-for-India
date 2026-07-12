import argparse
import json

import llm_client

# Neutral pole constructed via topic-matched rewriting rather than mined from Reddit,
# following Dual Optimal (persona - neutral, not persona - other_persona) and
# Ghandeharioun et al. 2024 (persona vs its own negation). Same content, no persona
# coloring — keeps the neutral pole topically matched to the positive pole.
NEUTRAL_PROMPT = """Rewrite the following household-related statement as a flat, neutral, third-person description of the same underlying action or request. Remove all emotional tone, authority, complaint, or persona-specific framing. Do not add new information. Keep it a single short sentence.

Statement: {text}

Respond with ONLY a JSON object, no preamble:
{{"neutral": "<flat neutral rewrite>"}}
"""


def neutralize(client, provider, model, text, max_chars=300):
    prompt = NEUTRAL_PROMPT.format(text=text[:max_chars])
    result = llm_client.call_llm_json(client, provider, prompt, model=model, max_tokens=150)
    if result is None:
        return None
    n = result.get("neutral", "").strip()
    return n if n else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-quotes", nargs="+", required=True,
                        help="Positive-pole quote files to neutralize")
    parser.add_argument("--provider", choices=["anthropic", "openai", "deepseek"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="neutral_baseline.json")
    args = parser.parse_args()

    client = llm_client.get_client(args.provider)

    source_texts = []
    for path in args.source_quotes:
        docs = json.load(open(path, encoding="utf-8"))["documents"]
        for d in docs:
            source_texts.append({"text": d["text"], "origin_source": path})

    print(f"Neutralizing {len(source_texts)} source quotes into flat register...")

    neutral = []
    for i, item in enumerate(source_texts):
        n = neutralize(client, args.provider, args.model, item["text"])
        if n:
            neutral.append({"text": n, "source": "neutral_baseline", "origin_text": item["text"]})
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(source_texts)}, produced {len(neutral)} neutral rewrites")

    print(f"\nProduced {len(neutral)} neutral-baseline statements.")
    json.dump({"documents": neutral}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
