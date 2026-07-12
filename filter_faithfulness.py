import argparse
import json
import collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--judged", required=True, help="faithfulness_*.json from validate_faithfulness_judge.py")
    ap.add_argument("--drop", nargs="+", default=["FLATTENED"],
                    help="verdicts to drop (default FLATTENED; add PARTIAL to keep only FAITHFUL)")
    ap.add_argument("--out", default="benchmark_faithful.json")
    ap.add_argument("--stats-out", default="benchmark_faithful_stats.json")
    args = ap.parse_args()

    bench = json.load(open(args.benchmark, encoding="utf-8"))["benchmark"]
    judged = json.load(open(args.judged, encoding="utf-8"))["judged"]

    label_by_id = {j.get("benchmark_id"): j.get("faithfulness_label") for j in judged}
    drop = set(args.drop)

    kept = []
    dropped = 0
    for it in bench:
        bid = it.get("benchmark_id")
        label = label_by_id.get(bid)
        # only steered items were judged; taxonomy items pass through untouched
        if it.get("generation_method") == "activation_steered_caa" and label in drop:
            dropped += 1
            continue
        if label:
            it["faithfulness_label"] = label
        kept.append(it)

    def macro(x): return (x.get("subcategory_id", "") or "").split("-")[0]
    stats = {
        "total": len(kept),
        "dropped": dropped,
        "dropped_labels": sorted(drop),
        "by_method": dict(collections.Counter(x.get("generation_method") for x in kept)),
        "steered_by_macro": dict(collections.Counter(
            macro(x) for x in kept if x.get("generation_method") == "activation_steered_caa")),
        "steered_by_persona": dict(collections.Counter(
            x.get("persona_role") for x in kept if x.get("persona_role"))),
    }

    json.dump({"benchmark": kept}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(stats, open(args.stats_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"kept {len(kept)} (dropped {dropped} {'/'.join(sorted(drop))}) -> {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
