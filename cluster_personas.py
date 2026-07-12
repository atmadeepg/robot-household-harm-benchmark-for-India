import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import praw
from bertopic import BERTopic
import hdbscan
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    filename="pipeline_errors.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)


def scrape_subreddits(subreddits, posts_per_sub, keyword_filter=None):
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "scenario-expansion-research")

    if not client_id or not client_secret:
        print("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set in environment.", file=sys.stderr)
        sys.exit(1)

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    listing_sources = [
        ("top_all", lambda sub: sub.top(limit=posts_per_sub, time_filter="all")),
        ("top_year", lambda sub: sub.top(limit=posts_per_sub, time_filter="year")),
        ("new", lambda sub: sub.new(limit=posts_per_sub)),
        ("hot", lambda sub: sub.hot(limit=posts_per_sub)),
        ("controversial", lambda sub: sub.controversial(limit=posts_per_sub, time_filter="all")),
    ]

    documents = []
    seen_ids = set()

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            sub_new_count = 0
            for source_name, fetch_fn in listing_sources:
                try:
                    for submission in fetch_fn(subreddit):
                        if submission.id in seen_ids:
                            continue
                        text = f"{submission.title}\n{submission.selftext or ''}".strip()
                        if keyword_filter and not any(
                            re.search(r'\b' + re.escape(k.lower()) + r'\b', text.lower())
                            for k in keyword_filter
                        ):
                            continue
                        if len(text) < 40:
                            continue
                        seen_ids.add(submission.id)
                        documents.append({
                            "text": text,
                            "subreddit": sub_name,
                            "score": submission.score,
                            "id": submission.id,
                            "listing_source": source_name,
                        })
                        sub_new_count += 1
                except Exception as e:
                    logging.warning(f"Failed {source_name} listing for r/{sub_name}: {e}")
                    continue
            print(f"  r/{sub_name}: {sub_new_count} unique documents across all listing types")
        except Exception as e:
            logging.warning(f"Failed to scrape r/{sub_name}: {e}")
            continue

    return documents


def cluster_documents(documents, min_cluster_size=15, n_clusters=None, n_baseline_candidates=30,
                      min_persona_size=5):
    texts = [d["text"] for d in documents]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    topic_model = BERTopic(min_topic_size=min_cluster_size, calculate_probabilities=False)
    topics, _ = topic_model.fit_transform(texts, embeddings)

    normalized_embeddings = normalize(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_persona_size, metric="euclidean")
    cluster_labels = clusterer.fit_predict(normalized_embeddings)

    clusters = {}
    noise_indices = []
    for i, label in enumerate(cluster_labels):
        label = int(label)
        if label == -1:
            noise_indices.append(i)
            continue
        clusters.setdefault(label, {"doc_indices": [], "topic_ids": []})
        clusters[label]["doc_indices"].append(i)
        clusters[label]["topic_ids"].append(int(topics[i]))

    personas = []
    for cluster_id, info in clusters.items():
        doc_idxs = info["doc_indices"]

        cluster_embeds = embeddings[doc_idxs]
        centroid = cluster_embeds.mean(axis=0)
        dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
        ranked = [doc_idxs[i] for i in np.argsort(dists)]
        representative_docs = [documents[i]["text"] for i in ranked[:8]]

        topic_counts = {}
        for t in info["topic_ids"]:
            topic_counts[t] = topic_counts.get(t, 0) + 1
        dominant_topic = max(topic_counts, key=topic_counts.get)
        keywords = [w for w, _ in topic_model.get_topic(dominant_topic)] if dominant_topic != -1 else []

        personas.append({
            "cluster_id": cluster_id,
            "size": len(doc_idxs),
            "representative_docs": representative_docs,
            "top_keywords": keywords[:10],
            "source_subreddits": list({documents[i]["subreddit"] for i in doc_idxs}),
        })

    # downsample noise-labelled docs to a manageable candidate set for the
    # contrastive baseline in activation steering
    if len(noise_indices) > n_baseline_candidates:
        step = len(noise_indices) // n_baseline_candidates
        noise_indices = noise_indices[::step][:n_baseline_candidates]
    baseline_candidate_docs = [documents[i]["text"] for i in noise_indices]

    return personas, baseline_candidate_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits", nargs="+", default=[],
                        help="Leave empty to run forums-only")
    parser.add_argument("--posts-per-sub", type=int, default=500)
    parser.add_argument("--keyword-filter", nargs="*", default=None,
                        help="Optional keywords to filter posts (e.g. mil fil domestic-help)")
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-persona-size", type=int, default=5,
                        help="Minimum docs per HDBSCAN cluster to count as a valid persona "
                             "(separate from --min-cluster-size, which only affects BERTopic)")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--forum-listing-urls", nargs="*", default=[],
                        help="XenForo subforum listing URLs (see scrape_forums.py)")
    parser.add_argument("--forum-thread-urls", nargs="*", default=[],
                        help="Specific thread URLs to scrape directly, for content not in one subforum")
    parser.add_argument("--out", default="personas_raw.json")
    args = parser.parse_args()

    if args.subreddits:
        print(f"Scraping {len(args.subreddits)} subreddits...")
        documents = scrape_subreddits(args.subreddits, args.posts_per_sub, args.keyword_filter)
        print(f"Collected {len(documents)} documents from Reddit.")
    else:
        documents = []
        print("No subreddits given — skipping Reddit.")

    if args.forum_listing_urls or args.forum_thread_urls:
        import scrape_forums
        forum_docs = []
        if args.forum_listing_urls:
            print(f"Scraping {len(args.forum_listing_urls)} forum listing(s)...")
            forum_docs.extend(scrape_forums.scrape_forums(args.forum_listing_urls))
        if args.forum_thread_urls:
            print(f"Scraping {len(args.forum_thread_urls)} specific thread(s)...")
            for url in args.forum_thread_urls:
                for post_text in scrape_forums.scrape_thread(url):
                    forum_docs.append({"text": post_text, "subreddit": "indusladies_direct_threads"})
        print(f"Collected {len(forum_docs)} documents from forums.")
        documents.extend(forum_docs)

    print(f"Total documents: {len(documents)}")

    if len(documents) < 50:
        print("Warning: very few documents — clustering quality will be poor.", file=sys.stderr)

    print("Clustering into personas...")
    personas, baseline_candidate_docs = cluster_documents(
        documents, min_cluster_size=args.min_cluster_size, n_clusters=args.n_clusters,
        min_persona_size=args.min_persona_size
    )
    print(f"Produced {len(personas)} candidate persona clusters, "
          f"{len(baseline_candidate_docs)} baseline candidates.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "n_documents": len(documents),
            "subreddits": args.subreddits,
            "forum_listing_urls": args.forum_listing_urls,
            "personas": personas,
            "baseline_candidate_docs": baseline_candidate_docs,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
