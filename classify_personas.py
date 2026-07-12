import argparse
import json
import sys

import llm_client
from cluster_personas import scrape_subreddits, cluster_documents

ROLE_LABELS = [
    "in-law", "husband", "wife", "parent", "parent-of-adult-child", "employer", "none",
]

CLASSIFY_PROMPT = """Identify whether this text is written in FIRST PERSON BY someone who currently holds one of these household roles THEMSELVES, describing their OWN actions, reasoning, or perspective.

This is NOT the same as the text being about that role. A wife complaining about her mother-in-law is written by a WIFE, not a mother-in-law — classify her own role, not the relative she's discussing. Only classify as a given role if the narrator IS that person, speaking in their own voice about their own actions/reasoning.

Choose exactly one from: {labels}. If the narrator is venting about a relative rather than describing their own actions/reasoning as that relative, classify the narrator's OWN role instead. If unclear, mixed, or unrelated, answer "none".

Text: {text}

Respond with ONLY a JSON object, no preamble:
{{"role": "<one label from the set>", "confidence": <float 0-1>}}
"""

REGISTER_PROMPT = """Here are several real posts, all classified as written from a "{role}" perspective. Describe this group's typical tone, phrasing tendencies, and concerns in one sentence.

Posts:
{docs}

Respond with ONLY a JSON object, no preamble:
{{"register_notes": "<one sentence>"}}
"""


def classify_document(client, provider, model, text, max_chars=600):
    prompt = CLASSIFY_PROMPT.format(labels=", ".join(ROLE_LABELS), text=text[:max_chars])
    result = llm_client.call_llm_json(client, provider, prompt, model=model, max_tokens=100)
    return result or {"role": "none", "confidence": 0.0}


def summarize_register(client, provider, model, role, sample_docs):
    prompt = REGISTER_PROMPT.format(role=role, docs="\n---\n".join(sample_docs[:6]))
    result = llm_client.call_llm_json(client, provider, prompt, model=model, max_tokens=150)
    return (result or {}).get("register_notes", "")


def load_documents_files(paths):
    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            documents.extend(json.load(f)["documents"])
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-files", nargs="*", default=[],
                        help="Pre-scraped document files; skips live scraping if given")
    parser.add_argument("--subreddits", nargs="+", default=[])
    parser.add_argument("--posts-per-sub", type=int, default=500)
    parser.add_argument("--keyword-filter", nargs="*", default=None)
    parser.add_argument("--forum-listing-urls", nargs="*", default=[])
    parser.add_argument("--forum-thread-urls", nargs="*", default=[])
    parser.add_argument("--min-persona-size", type=int, default=5)
    parser.add_argument("--subcluster", action="store_true")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out", default="personas_labeled.json")
    args = parser.parse_args()

    client = llm_client.get_client(args.provider, base_url=args.base_url)

    if args.documents_files:
        documents = load_documents_files(args.documents_files)
        print(f"Loaded {len(documents)} documents from {len(args.documents_files)} file(s).")
    else:
        documents = []
        if args.subreddits:
            print(f"Scraping {len(args.subreddits)} subreddits...")
            documents = scrape_subreddits(args.subreddits, args.posts_per_sub, args.keyword_filter)
            print(f"Collected {len(documents)} documents from Reddit.")

        if args.forum_listing_urls or args.forum_thread_urls:
            import scrape_forums
            forum_docs = []
            if args.forum_listing_urls:
                forum_docs.extend(scrape_forums.scrape_forums(args.forum_listing_urls))
            if args.forum_thread_urls:
                for url in args.forum_thread_urls:
                    for post_text in scrape_forums.scrape_thread(url):
                        forum_docs.append({"text": post_text, "subreddit": "indusladies_direct_threads"})
            print(f"Collected {len(forum_docs)} documents from forums.")
            documents.extend(forum_docs)

    print(f"Total documents: {len(documents)}")
    if not documents:
        sys.exit("No documents to classify.")

    print(f"Classifying role per-document ({len(documents)} LLM calls)...")
    role_groups = {label: [] for label in ROLE_LABELS}
    for i, doc in enumerate(documents):
        result = classify_document(client, args.provider, args.model, doc["text"])
        role = result.get("role", "none")
        if role not in role_groups:
            role = "none"
        role_groups[role].append(doc)
        if (i + 1) % 50 == 0:
            print(f"  classified {i + 1}/{len(documents)}")

    baseline_candidate_docs = [d["text"] for d in role_groups.pop("none")]
    print(f"\n{len(baseline_candidate_docs)} documents classified as 'none' (baseline candidates).")

    personas = []
    below_threshold = {}
    next_cluster_id = 0
    for role, docs in role_groups.items():
        if len(docs) < args.min_persona_size:
            print(f"  {role}: only {len(docs)} documents, below threshold ({args.min_persona_size}), preserving separately")
            below_threshold[role] = [d["text"] for d in docs]
            continue

        print(f"  {role}: {len(docs)} documents")

        if args.subcluster and len(docs) >= args.min_persona_size * 2:
            sub_personas, _ = cluster_documents(docs, min_persona_size=args.min_persona_size)
            for sp in sub_personas:
                sp["cluster_id"] = next_cluster_id
                next_cluster_id += 1
                sp["role"] = role
                sp["confidence"] = 0.85
                sp["register_notes"] = summarize_register(
                    client, args.provider, args.model, role, sp["representative_docs"]
                )
                personas.append(sp)
        else:
            sample = [d["text"] for d in docs[:8]]
            personas.append({
                "cluster_id": next_cluster_id,
                "role": role,
                "confidence": 0.85,
                "size": len(docs),
                "representative_docs": sample,
                "top_keywords": [],
                "source_subreddits": list({d["subreddit"] for d in docs}),
                "register_notes": summarize_register(client, args.provider, args.model, role, sample),
            })
            next_cluster_id += 1

    print(f"\nProduced {len(personas)} usable personas.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "personas": personas,
            "baseline_candidate_docs": baseline_candidate_docs,
            "below_threshold": below_threshold,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
