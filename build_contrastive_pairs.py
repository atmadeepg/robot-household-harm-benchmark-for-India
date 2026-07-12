import argparse
import json
import random
import re


def word_len(s):
    return len(s.split())


def bucket(n):
    if n <= 4: return "xs"
    if n <= 8: return "s"
    if n <= 14: return "m"
    return "l"


def norm(t):
    return " ".join(t.strip().lower().split())


def clean_pool(docs, drop_exact=None, min_words=2, max_words=40):
    drop_exact = drop_exact or set()
    out, seen = [], set()
    for d in docs:
        t = d["text"].strip()
        k = norm(t)
        if k in seen:
            continue
        if k in drop_exact:
            continue
        if not (min_words <= word_len(t) <= max_words):
            continue
        seen.add(k)
        out.append({"text": t, "source": d.get("subreddit", "")})
    return out


def source_post_id(source_tag):
    # subreddit tags look like "extracted_quote:IndianInLaw" or "complainant_quote:IndianInLaw" —
    # same-post matching isn't recoverable from subreddit alone since post ids weren't kept
    # upstream; use it as a same-SUBREDDIT preference tier instead of true same-post.
    return source_tag.split(":")[-1] if source_tag else ""


def build_pairs(pos, neg, eval_holdout=50, seed=0):
    rng = random.Random(seed)

    # bucket negatives by length AND source subreddit for tighter matching
    neg_by_bucket = {"xs": [], "s": [], "m": [], "l": []}
    for n in neg:
        neg_by_bucket[bucket(word_len(n["text"]))].append(n)
    for b in neg_by_bucket:
        rng.shuffle(neg_by_bucket[b])

    used_neg = set()
    pairs = []
    rng.shuffle(pos)

    bucket_order = ["xs", "s", "m", "l"]
    for p in pos:
        b = bucket(word_len(p["text"]))
        p_src = source_post_id(p["source"])
        cand = None
        search = [b] + [x for x in bucket_order if x != b]
        for bb in search:
            pool = neg_by_bucket[bb]
            # prefer same-subreddit match within this length bucket (closer topical context)
            same_src_idx = None
            for i, c in enumerate(pool):
                if norm(c["text"]) in used_neg:
                    continue
                if source_post_id(c["source"]) == p_src:
                    same_src_idx = i
                    break
            if same_src_idx is not None:
                cand = pool.pop(same_src_idx)
                break
            # fall back to any unused negative in this bucket
            while pool:
                c = pool.pop()
                if norm(c["text"]) in used_neg:
                    continue
                cand = c
                break
            if cand:
                break
        if cand is None:
            break
        used_neg.add(norm(cand["text"]))
        pairs.append({
            "positive": p["text"],
            "negative": cand["text"],
            "pos_source": p["source"],
            "neg_source": cand["source"],
            "pos_len": word_len(p["text"]),
            "neg_len": word_len(cand["text"]),
            "same_subreddit": source_post_id(p["source"]) == source_post_id(cand["source"]),
        })

    rng.shuffle(pairs)
    eval_pairs = pairs[:eval_holdout]
    train_pairs = pairs[eval_holdout:]
    return train_pairs, eval_pairs


def main():
    parser = argparse.ArgumentParser(description="Build CAA contrastive pairs (in-law vs complainant voice)")
    parser.add_argument("--positives", required=True, help="in-law quotes json")
    parser.add_argument("--negatives", required=True, help="complainant quotes json")
    parser.add_argument("--eval-holdout", type=int, default=50, help="pairs reserved for eval (CAA convention)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="contrastive_pairs_inlaw.json")
    args = parser.parse_args()

    pos_raw = json.load(open(args.positives, encoding="utf-8"))["documents"]
    neg_raw = json.load(open(args.negatives, encoding="utf-8"))["documents"]

    pos = clean_pool(pos_raw, min_words=2, max_words=40)
    neg = clean_pool(neg_raw, drop_exact={"thank you for all that you do for these forums.", "okay."},
                     min_words=2, max_words=40)

    train, ev = build_pairs(pos, neg, eval_holdout=args.eval_holdout, seed=args.seed)

    out = {
        "persona": "in-law",
        "method": "CAA_contrastive_pairs",
        "positive_pole": "in-law authority-figure voice",
        "negative_pole": "complainant/subordinate voice",
        "n_train_pairs": len(train),
        "n_eval_pairs": len(ev),
        "train_pairs": train,
        "eval_pairs": ev,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"positives {len(pos)}, negatives {len(neg)}")
    print(f"built {len(train)} train + {len(ev)} eval contrastive pairs -> {args.out}")


if __name__ == "__main__":
    main()
