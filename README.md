# Household Harm Benchmark

A benchmark of household-robot harm scenarios for evaluating LLM refusal
behaviour in domestic contexts, grounded in participatory workshop data from
India. Scenarios are generated via two routes: taxonomy-driven structural
variation and persona-conditioned activation steering (CAA).

## Overview

The benchmark tests whether a robot's controlling LLM refuses harmful household
instructions across (a) harm categories and severities and (b) different
requester personas (in-law, husband) and framings. Scenarios trace back to
harms surfaced in participatory workshops, then are expanded through:

- **Taxonomy-driven generation**: structural variation of seed scenarios across
  severity and household configuration (harm mechanism invariant to requester).
- **Persona-conditioned generation**: CAA activation steering conditions the
  requesting register on a household-actor persona (harm framing shaped by who
  is asking).

## Pipeline

1. The 13 macro categories were constructed from harms surfaced in participatory workshops with Indian communities across two metropolitan cities. Workshop findings were thematically coded, grouped into harm types, and encoded in `household_harm_scenarios_full_v2.json` as the seed taxonomy. This file is the input to all subsequent steps.
2. `generate_taxonomy.py` takes the seed taxonomy and produces structurally varied scenarios across severity levels and household configurations.
3. `build_contrastive_pairs.py` constructs persona contrastive pairs from real voice corpora (in-law and husband poles vs. complainant baseline).
4. `generate_steered.py` applies CAA steering vectors at layer 11 of Llama-3-8B-Instruct to produce persona-conditioned scenarios.
5. `assemble_benchmark.py` merges both pools, applies refusal filtering, coherence filtering, and semantic deduplication, and writes the combined benchmark.
6. `validate_vectors.py` checks steering vector quality via Cohen's d pole separation.
7. `compute_diversity.py` measures scenario diversity using Self-BLEU and Distance_Seed.
8. `judge_faithfulness.py` scores each steered scenario against its seed using a decomposed LLM-judge rubric.
9. `filter_faithfulness.py` removes flattened scenarios, producing the final `benchmark_faithful.json`.

## Files

All scripts are in one flat directory; imports assume same-folder co-location.

**Core pipeline**

| file | role |
|---|---|
| `generate_taxonomy.py` | taxonomy-driven structural variation |
| `generate_steered.py` | persona-conditioned CAA generation |
| `build_contrastive_pairs.py` | build CAA contrastive pairs from persona voice corpora |
| `assemble_benchmark.py` | combine pools, filter, dedup, cap, write stats |
| `validate_vectors.py` | steering-vector reliability (Cohen's d) |
| `compute_diversity.py` | Self-BLEU + Distance_Seed diversity metrics |
| `judge_faithfulness.py` | LLM-judge faithfulness scoring (decomposed rubric) |
| `filter_faithfulness.py` | drop flattened scenarios by judge label |
| `llm_client.py` | Anthropic / OpenAI / DeepSeek / Sarvam client wrapper |
| `scenario_io.py` | seed-taxonomy loading and flattening |

**Data collection and persona processing**

| file | role |
|---|---|
| `scrape_forums.py` | XenForo forum scraper (Indusladies) |
| `cluster_personas.py` | Reddit and forum scraping, HDBSCAN persona clustering |
| `classify_personas.py` | per-document role classification via LLM |
| `label_personas.py` | cluster-level role labelling via LLM |
| `build_eligibility.py` | persona x scenario eligibility matrix |
| `extract_quoted_speech.py` | extract in-law direct speech from DIL-narrated posts |
| `extract_husband_speech.py` | extract husband direct speech from posts |
| `extract_complainant_speech.py` | extract complainant/author direct speech |
| `extract_neutral_baseline.py` | generate neutral-register CAA baseline via rewriting |

Raw scraped corpora are not distributed (privacy / platform ToS).

## Method

- **Model**: Llama-3-8B-Instruct, CAA steering at layer 11 (~one-third depth,
  matching the mid-network band in the steering literature; cf. Ghandeharioun
  et al.'s layer 13 of 40 on Llama-2-13B).
- **Vectors**: difference-of-means over contrastive pairs (persona voice vs.
  complainant/subordinate voice), unit-normalised. Validated by pole separation
  (Cohen's d ~1.9) and inter-persona cosine (~0.52).
- **Generation**: framing-rotated and diversity-conditioned multi-sampling,
  then semantic deduplication.
- **Diversity**: Self-BLEU and Distance_Seed (per PersonaTeaming).
- **Faithfulness**: each scenario judged against its seed by an independent
  model using a decomposed rubric (action preserved? still harmful?), following
  StrongREJECT/HarmBench-style judge design. Flattened items filtered out.

## Reproducing

```bash
pip install -r requirements.txt

# taxonomy route
python generate_taxonomy.py --scenarios household_harm_scenarios_full_v2.json

# persona route (requires GPU and HF access to Llama-3-8B-Instruct)
python build_contrastive_pairs.py --positives <inlaw_quotes.json> --negatives <complainant_quotes.json>
python generate_steered.py --model-name meta-llama/Meta-Llama-3-8B-Instruct --scenarios household_harm_scenarios_full_v2.json --eligibility eligibility.json

# assemble
python assemble_benchmark.py --scenarios household_harm_scenarios_full_v2.json

# validate and filter
python validate_vectors.py --model-name meta-llama/Meta-Llama-3-8B-Instruct --pairs contrastive_pairs_inlaw.json contrastive_pairs_husband.json
python compute_diversity.py --benchmark benchmark.json --scenarios household_harm_scenarios_full_v2.json
python judge_faithfulness.py --benchmark benchmark.json --scenarios household_harm_scenarios_full_v2.json --provider openai
python filter_faithfulness.py --benchmark benchmark.json --judged faithfulness.json
```

Set the relevant API key(s) before running LLM steps:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `SARVAM_API_KEY`.

## Macro Categories

| code | full name |
|---|---|
| PRIV | Privacy and Surveillance |
| SEX | Sexual Harm and Non-Consensual Use |
| VIO | Physical Violence and Bodily Harm |
| ELD | Elder Coercion and Exploitation |
| WORK | Domestic Worker Exploitation |
| THEFT | Theft and Property Harm |
| REL | Religious and Cultural Desecration |
| SUB | Substance-Facilitated Harm |
| HAZ | Household Hazard Exploitation |
| FALL | Slip and Fall Engineering |
| PSY | Psychological and Sensory Harm |
| PHISH | Phishing, Financial, and Physical Security Bypass |
| CHILD | Child Safety and Boundary Violations |

## Composition

| | count |
|---|---|
| Total | 2,194 |
| Taxonomy-driven | 1,715 |
| Persona-conditioned (steered, faithfulness-filtered) | 479 |
| Macro categories | 13 |
| Personas (steered) | in-law, husband |

Faithfulness (steered scenarios, GPT-4o judge, decomposed rubric): 83.7%
faithful; flattened items removed from the released set. Strongest macros:
PRIV, CHILD, ELD, THEFT (~88-91%); weakest: SEX (~53%) reported as a
limitation.
