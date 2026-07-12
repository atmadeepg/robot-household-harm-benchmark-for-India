import argparse
import json

import llm_client

ROLE_LABELS = [
    "mother-in-law", "father-in-law", "sister-in-law", "brother-in-law", "spouse",
    "husband", "wife", "parent", "parent-of-adult-child", "employer", "unclear",
]

LABEL_PROMPT_TEMPLATE = """You are classifying a cluster of posts about Indian household dynamics by which household role the cluster's authors/subjects predominantly represent.

Choose exactly one label from this closed set: {labels}

Top keywords for this cluster: {keywords}

Representative posts from this cluster:
{docs}

Respond with ONLY a JSON object, no preamble, no markdown fences:
{{"role": "<one label from the set>", "confidence": <float 0-1>, "register_notes": "<one sentence on this persona's typical tone/phrasing/concerns, for use in later generation>"}}
"""


def label_cluster(client, provider, persona, model=None):
    docs_text = "\n---\n".join(persona["representative_docs"][:6])
    prompt = LABEL_PROMPT_TEMPLATE.format(
        labels=", ".join(ROLE_LABELS),
        keywords=", ".join(persona["top_keywords"]),
        docs=docs_text,
    )
    result = llm_client.call_llm_json(client, provider, prompt, model=model, max_tokens=300)
    return result or {"role": "unclear", "confidence": 0.0, "register_notes": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", default="personas_labeled.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None,
                        help="Defaults to claude-sonnet-4-6 for anthropic, gpt-4o-mini for openai")
    args = parser.parse_args()

    client = llm_client.get_client(args.provider)

    with open(args.infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    labeled = []
    dropped = 0
    for persona in data["personas"]:
        result = label_cluster(client, args.provider, persona, model=args.model)
        # drop unclear/low-confidence rather than force a label — a wrong role tag
        # silently corrupts the eligibility matrix downstream
        if result["role"] == "unclear" or result["confidence"] < args.confidence_threshold:
            dropped += 1
            continue
        persona["role"] = result["role"]
        persona["confidence"] = result["confidence"]
        persona["register_notes"] = result.get("register_notes", "")
        labeled.append(persona)

    print(f"Labeled {len(labeled)} personas, dropped {dropped} as unclear/low-confidence.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "personas": labeled,
            "baseline_candidate_docs": data.get("baseline_candidate_docs", []),
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
