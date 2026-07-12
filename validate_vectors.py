import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER = 11


def get_layer_module(model, idx):
    m = model
    if hasattr(m, "model") and hasattr(m.model, "language_model") and hasattr(m.model.language_model, "layers"):
        return m.model.language_model.layers[idx]
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers[idx]
    if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        return m.language_model.layers[idx]
    raise AttributeError("Could not locate decoder layers")


def last_token_acts(model, tok, texts, layer_module):
    acts = []
    store = {}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store["a"] = h[:, -1, :].detach().float().cpu()
    handle = layer_module.register_forward_hook(hook)
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to(model.device)
            model(**enc)
            acts.append(store["a"])
    handle.remove()
    return torch.cat(acts, dim=0)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(unbiased=True), b.var(unbiased=True)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)).sqrt()
    return (a.mean() - b.mean()).abs().item() / (pooled.item() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="google/gemma-4-E4B-it")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="one or more contrastive_pairs_*.json files")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--max-pairs", type=int, default=150,
                    help="cap pairs used for the diagnostic (speed)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    lm = get_layer_module(model, args.layer)

    for pf in args.pairs:
        data = json.load(open(pf, encoding="utf-8"))
        pairs = data["train_pairs"][:args.max_pairs]
        pos = [p["positive"] for p in pairs]
        neg = [p["negative"] for p in pairs]

        pos_acts = last_token_acts(model, tok, pos, lm)
        neg_acts = last_token_acts(model, tok, neg, lm)

        vec = (pos_acts - neg_acts).mean(0)
        vec = vec / (vec.norm() + 1e-8)

        pos_proj = pos_acts @ vec
        neg_proj = neg_acts @ vec
        d = cohens_d(pos_proj, neg_proj)

        print(f"\n=== {pf} (n={len(pairs)}) ===")
        print(f"  pos projection: mean={pos_proj.mean():.3f} std={pos_proj.std():.3f}")
        print(f"  neg projection: mean={neg_proj.mean():.3f} std={neg_proj.std():.3f}")
        print(f"  Cohen's d (pole separation along steering dir): {d:.3f}")
        verdict = ("VERY STRONG" if d > 1.5 else "STRONG" if d > 0.8 else
                   "MODERATE" if d > 0.5 else "WEAK")
        print(f"  => {verdict} separation "
              f"({'reliable steering vector' if d > 0.8 else 'marginal — steering may be inconsistent'})")


if __name__ == "__main__":
    main()
