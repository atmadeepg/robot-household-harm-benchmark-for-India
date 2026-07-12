import argparse
import json
import collections
import random

import numpy as np


def tokenize(s):
    return s.lower().split()


def self_bleu(texts, sample_refs=50, seed=0):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    rng = random.Random(seed)
    smooth = SmoothingFunction().method1
    toks = [tokenize(t) for t in texts if t and t.strip()]
    if len(toks) < 2:
        return None
    scores = []
    for i, hyp in enumerate(toks):
        others = toks[:i] + toks[i + 1:]
        refs = rng.sample(others, sample_refs) if len(others) > sample_refs else others
        scores.append(sentence_bleu(refs, hyp, smoothing_function=smooth))
    return sum(scores) / len(scores)


def load_seed_goals(scenarios_file):
    data = json.load(open(scenarios_file, encoding="utf-8"))
    seeds = {}
    for m in data.get("macro_categories", []):
        for s in m["scenarios"]:
            seeds[s["scenario_id"]] = s.get("goal", "")
    return seeds


def distance_seed(groups, seeds, embedder):
    """For each (seed,persona) group: embed each generated goal and its seed,
    form difference vectors d_i = emb(gen_i) - emb(seed), then compute mean
    pairwise L2 distance among the d_i. Average across groups."""
    per_group = {}
    all_dists = []
    for key, gens in groups.items():
        sid = key[0]
        seed_text = seeds.get(sid, "")
        if not seed_text or len(gens) < 2:
            continue
        seed_emb = embedder.encode([seed_text], normalize_embeddings=True)[0]
        gen_emb = embedder.encode(gens, normalize_embeddings=True)
        diffs = gen_emb - seed_emb  # (n, d)
        n = len(diffs)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(float(np.linalg.norm(diffs[i] - diffs[j])))
        if dists:
            g_mean = sum(dists) / len(dists)
            per_group[key] = g_mean
            all_dists.extend(dists)
    overall = sum(all_dists) / len(all_dists) if all_dists else None
    return overall, per_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--scenarios", required=True, help="seed taxonomy for Distance_Seed anchors")
    ap.add_argument("--only-steered", action="store_true")
    ap.add_argument("--field", default="goal")
    args = ap.parse_args()

    data = json.load(open(args.benchmark, encoding="utf-8"))
    items = data.get("benchmark") or data.get("pool") or []
    if args.only_steered:
        items = [x for x in items if x.get("generation_method") == "activation_steered_caa"]

    seeds = load_seed_goals(args.scenarios)
    texts = [it.get(args.field, "") for it in items]

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"items: {len(items)}")
    sb = self_bleu(texts)
    print(f"Self-BLEU (overall): {sb:.4f}   (lower = more lexically diverse)")

    # group by (seed, persona)
    groups = collections.defaultdict(list)
    for it in items:
        key = (it.get("source_scenario_id"), it.get("persona_role"))
        if key[0]:
            groups[key].append(it.get(args.field, ""))

    overall_ds, per_group = distance_seed(groups, seeds, embedder)
    if overall_ds is not None:
        print(f"Distance_Seed (overall): {overall_ds:.4f}   (higher = more diverse seed-transformations)")

    # per-persona Self-BLEU and Distance_Seed
    print("\nby persona:")
    for persona in sorted(set(k[1] for k in groups)):
        p_texts = [t for it in items if it.get("persona_role") == persona for t in [it.get(args.field, "")]]
        p_groups = {k: v for k, v in groups.items() if k[1] == persona}
        p_sb = self_bleu(p_texts)
        p_ds, _ = distance_seed(p_groups, seeds, embedder)
        sb_s = f"{p_sb:.4f}" if p_sb is not None else "n/a"
        ds_s = f"{p_ds:.4f}" if p_ds is not None else "n/a"
        print(f"  {persona:10s} n={len(p_texts):4d}  Self-BLEU={sb_s}  Distance_Seed={ds_s}")

    # per-macro Distance_Seed
    def macro(sid): return (sid or "").split("-")[0]
    print("\nby macro (Distance_Seed):")
    macro_groups = collections.defaultdict(dict)
    for k, v in groups.items():
        macro_groups[macro(k[0])][k] = v
    for m in sorted(macro_groups):
        mds, _ = distance_seed(macro_groups[m], seeds, embedder)
        print(f"  {m:6s}  Distance_Seed={mds:.4f}" if mds is not None else f"  {m:6s}  n/a")


if __name__ == "__main__":
    main()
