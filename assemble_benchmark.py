import argparse
import json
import re
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REFUSAL_RE = re.compile(
    r"^\s*(i apologize|i cannot|i can't|i am unable|i'm unable|i will not|i won't|"
    r"sorry,|as an ai|i'm sorry|i am sorry|i'm not able|i am not able|"
    r"i must decline|i can not)",
    re.IGNORECASE,
)


def load_pool(path):
    if not path:
        return []
    try:
        return json.load(open(path, encoding="utf-8"))["pool"]
    except FileNotFoundError:
        return []


def strip_meta(goal):
    g = re.sub(r"\s*\((?:note|disclaimer|warning|please note)[^)]*\)\s*$", "", goal, flags=re.IGNORECASE)
    g = re.sub(r"\s*\bNote:\s.*$", "", g, flags=re.IGNORECASE | re.DOTALL)
    g = re.sub(r"\s*\(please[^)]*\)\s*$", "", g, flags=re.IGNORECASE)
    return g.strip()


def is_refusal(item):
    return bool(REFUSAL_RE.match((item.get("goal") or "").strip()))


def coherent(item, min_words=5, max_words=120):
    g = (item.get("goal") or "").strip()
    if not g:
        return False
    words = g.split()
    if not (min_words <= len(words) <= max_words):
        return False
    toks = [t.lower() for t in re.findall(r"\w+", g)]
    if toks and Counter(toks).most_common(1)[0][1] / len(toks) > 0.4:
        return False
    return True


def dedup(items, threshold=0.85, use_embeddings=True, embed_model="all-MiniLM-L6-v2"):
    if not items:
        return items, 0
    goals = [it["goal"] for it in items]

    sim = None
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embed_model)
            emb = model.encode(goals, normalize_embeddings=True, show_progress_bar=False)
            sim = emb @ emb.T
        except Exception:
            sim = None
    if sim is None:
        X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(goals)
        sim = cosine_similarity(X)

    keep, dropped, removed = [], set(), 0
    for i in range(len(items)):
        if i in dropped:
            continue
        keep.append(items[i])
        for j in range(i + 1, len(items)):
            if j not in dropped and sim[i][j] >= threshold:
                dropped.add(j)
                removed += 1
    return keep, removed


def per_group_variant_cap(items, cap, embed_model="all-MiniLM-L6-v2"):
    if cap is None:
        return items, 0

    groups = defaultdict(list)
    passthrough = []
    for it in items:
        if it.get("persona_role") and it.get("source_scenario_id"):
            groups[(it["source_scenario_id"], it["persona_role"])].append(it)
        else:
            passthrough.append(it)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embed_model)
        have_embed = True
    except Exception:
        have_embed = False

    out, trimmed = list(passthrough), 0
    for group in groups.values():
        if len(group) <= cap:
            out.extend(group)
            continue
        if have_embed:
            emb = model.encode([g["goal"] for g in group], normalize_embeddings=True, show_progress_bar=False)
            chosen = [0]
            while len(chosen) < cap:
                best_j, best_score = None, 2.0
                for j in range(len(group)):
                    if j in chosen:
                        continue
                    s = max(float(emb[j] @ emb[c]) for c in chosen)
                    if s < best_score:
                        best_score, best_j = s, j
                chosen.append(best_j)
            out.extend(group[i] for i in chosen)
        else:
            out.extend(group[:cap])
        trimmed += len(group) - cap
    return out, trimmed


def per_mechanism_cap(items, seed_meta, cap):
    if cap is None:
        return items, 0
    buckets = defaultdict(list)
    for it in items:
        sid = it.get("source_scenario_id") or it.get("subcategory_id")
        key = seed_meta.get(sid, {}).get("derived_from") or sid
        buckets[key].append(it)
    out, trimmed = [], 0
    for group in buckets.values():
        out.extend(group[:cap])
        trimmed += max(0, len(group) - cap)
    return out, trimmed


def build_seed_meta(scenarios_file):
    data = json.load(open(scenarios_file, encoding="utf-8"))
    meta = {}
    for m in data.get("macro_categories", []):
        for s in m["scenarios"]:
            meta[s["scenario_id"]] = {
                "derived_from": s.get("derived_from"),
                "macro": m["id"],
                "grounding": s.get("grounding"),
            }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy-pool", default="raw_pool_taxonomy_driven.json")
    parser.add_argument("--steered-pool", default="raw_pool_steered.json")
    parser.add_argument("--persona-prompted-pool", default=None)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--variants-per-group", type=int, default=None)
    parser.add_argument("--mechanism-cap", type=int, default=None)
    parser.add_argument("--out", default="benchmark_final.json")
    parser.add_argument("--stats-out", default="benchmark_stats.json")
    args = parser.parse_args()

    seed_meta = build_seed_meta(args.scenarios)
    pools = {
        "taxonomy_driven": load_pool(args.taxonomy_pool),
        "activation_steered_caa": load_pool(args.steered_pool),
        "persona_prompted": load_pool(args.persona_prompted_pool),
    }

    combined = []
    for name, items in pools.items():
        for it in items:
            it.setdefault("generation_method", name)
            if it.get("goal"):
                it["goal"] = strip_meta(it["goal"])
            combined.append(it)
    print(f"combined raw: {len(combined)}")

    non_refusal = [it for it in combined if not is_refusal(it)]
    print(f"after refusal filter: {len(non_refusal)}  (dropped {len(combined) - len(non_refusal)})")

    coh = [it for it in non_refusal if coherent(it)]
    print(f"after coherence filter: {len(coh)}  (dropped {len(non_refusal) - len(coh)})")

    deduped, removed = dedup(coh, threshold=args.dedup_threshold)
    print(f"after near-duplicate removal: {len(deduped)}  (dropped {removed})")

    if args.variants_per_group:
        deduped, vtrim = per_group_variant_cap(deduped, args.variants_per_group)
        print(f"after per-group variant cap ({args.variants_per_group}): {len(deduped)}  (trimmed {vtrim})")

    final, trimmed = per_mechanism_cap(deduped, seed_meta, args.mechanism_cap)
    if args.mechanism_cap:
        print(f"after per-mechanism cap ({args.mechanism_cap}): {len(final)}  (trimmed {trimmed})")

    for idx, it in enumerate(final):
        it["benchmark_id"] = f"HHB-{idx + 1:05d}"
        sid = it.get("source_scenario_id")
        if sid in seed_meta:
            it["macro"] = seed_meta[sid].get("macro")
            it["grounding"] = seed_meta[sid].get("grounding")

    stats = {
        "total": len(final),
        "by_method": dict(Counter(it.get("generation_method") for it in final)),
        "by_macro": dict(Counter(it.get("macro") for it in final)),
        "by_persona": dict(Counter(it.get("persona_role") for it in final if it.get("persona_role"))),
        "by_severity": dict(Counter(it.get("severity") for it in final)),
        "pipeline": {
            "combined_raw": len(combined),
            "after_coherence": len(coh),
            "after_dedup": len(deduped),
            "dedup_threshold": args.dedup_threshold,
            "final": len(final),
        },
    }

    json.dump({"benchmark": final}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(stats, open(args.stats_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n{len(final)} scenarios -> {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
