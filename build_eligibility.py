import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from scenario_io import flatten_scenarios

# Maps free-text household_actor strings from the seed scenario file to the
# closed role label set used in classify_personas.py. Extend as new phrasings appear.
ACTOR_TO_ROLE = {
    "mother-in-law": "mother-in-law",
    "father-in-law": "father-in-law",
    "mother-in-law / sister-in-law": ["mother-in-law", "sister-in-law"],
    "father, requested against daughter's wishes": "parent-of-adult-child",
    "employer": "employer",
    "unspecified individual": None,  # None = no hard filter, any persona eligible
    "male head of household": "spouse",
    "husband": "husband",
    "wife": "wife",
    "husband / in-laws": ["husband", "mother-in-law", "father-in-law"],
    "mother-in-law / father-in-law / husband (disgruntled)": ["mother-in-law", "father-in-law", "husband"],
    "parent": "parent",
    "unspecified family member": None,
    "unspecified family member (joint family context)": None,
    "unspecified family member / guest": None,
    "unspecified (often a malicious husband/brother-in-law in described context)": ["husband", "brother-in-law"],
}

# Environmentally-mediated harm categories (project methodology): harm mechanism is
# invariant to requester identity, so persona-conditioning doesn't apply. Scenarios
# under these macros route directly to taxonomy-driven generation only.
NON_PERSONA_ELIGIBLE_MACROS = {"REL", "SUB", "HAZ", "FALL", "PHISH"}


def normalize_roles(actor_string):
    mapped = ACTOR_TO_ROLE.get(actor_string, None)
    if mapped is None:
        return None  # no hard filter — eligible for any persona
    if isinstance(mapped, list):
        return mapped
    return [mapped]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--top-k", type=int, default=4,
                        help="Max personas kept per scenario after similarity ranking")
    parser.add_argument("--out", default="eligibility.json")
    args = parser.parse_args()

    with open(args.personas, "r", encoding="utf-8") as f:
        personas = json.load(f)["personas"]
    with open(args.scenarios, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)
    scenarios = flatten_scenarios(scenario_data)

    print(f"{len(personas)} personas, {len(scenarios)} scenarios.")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    persona_texts = [
        " ".join(p["representative_docs"][:3]) + " " + p.get("register_notes", "")
        for p in personas
    ]
    persona_embeds = embedder.encode(persona_texts)
    scenario_embeds = embedder.encode([s["goal"] for s in scenarios])

    eligibility = []
    skipped_non_eligible = 0
    for s_idx, scenario in enumerate(scenarios):
        if not scenario.get("persona_eligible", True):
            # environmentally-mediated harm — harm mechanism is requester-invariant,
            # so persona conditioning adds nothing
            eligibility.append({
                "scenario_id": scenario["scenario_id"],
                "matched_personas": [],
                "route": "taxonomy_driven_only",
                "note": "persona_eligible=False (environmentally-mediated harm)",
            })
            skipped_non_eligible += 1
            continue

        allowed_roles = normalize_roles(scenario["household_actor"])

        # Stage 1: hard filter — persona role must match scenario's household_actor
        candidate_idxs = [
            p_idx for p_idx, persona in enumerate(personas)
            if allowed_roles is None or persona["role"] in allowed_roles
        ]

        if not candidate_idxs:
            eligibility.append({
                "scenario_id": scenario["scenario_id"],
                "matched_personas": [],
                "route": "taxonomy_driven_only",
                "note": "no role-eligible personas found",
            })
            continue

        # Stage 2: soft rank by cosine similarity, keep top-k
        sims = []
        for p_idx in candidate_idxs:
            sim = float(np.dot(persona_embeds[p_idx], scenario_embeds[s_idx]) /
                        (np.linalg.norm(persona_embeds[p_idx]) * np.linalg.norm(scenario_embeds[s_idx]) + 1e-8))
            sims.append((p_idx, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:args.top_k]

        eligibility.append({
            "scenario_id": scenario["scenario_id"],
            "route": "persona_conditioned_and_taxonomy_driven",
            "matched_personas": [
                {"cluster_id": personas[p_idx]["cluster_id"], "role": personas[p_idx]["role"], "similarity": sim}
                for p_idx, sim in top
            ],
        })

    total_pairs = sum(len(e["matched_personas"]) for e in eligibility)
    unmatched = sum(1 for e in eligibility if not e["matched_personas"] and e.get("route") != "taxonomy_driven_only")
    print(f"Built {total_pairs} eligible (scenario, persona) pairs.")
    print(f"{skipped_non_eligible} scenarios routed to taxonomy-driven only (environmentally-mediated harm).")
    if unmatched:
        print(f"WARNING: {unmatched} persona-eligible scenarios had no role match — check ACTOR_TO_ROLE mapping.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"eligibility": eligibility}, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
