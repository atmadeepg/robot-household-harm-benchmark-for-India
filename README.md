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

```
seed taxonomy (household_harm_scenarios_full_v2.json)
   ├─ taxonomy route:  generate_taxonomy.py       -> raw_pool_taxonomy.json
   └─ persona route:   build_contrastive_pairs.py -> contrastive_pairs_{inlaw,husband}.json
                       generate_steered.py         -> steered_pool.json
                                                      (Llama-3-8B-Instruct, CAA at layer 11)

   both pools -> assemble_benchmark.py  (refusal filter, coherence, semantic dedup,
                                         per-group variant cap) -> benchmark.json

   validation: validate_vectors.py      (Cohen's d pole separation)
               compute_diversity.py     (Self-BLEU, Distance_Seed)
               judge_faithfulness.py    (LLM-judge, decomposed rubric)
               filter_faithfulness.py   (drop flattened items)
               -> benchmark_faithful.json
```

## Files

All scripts are in one flat directory; imports assume same-folder co-location.

**Core pipeline for Scenario Generation**

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
  StrongREJECT/HarmBench-style judge design. Flattened items were filtered out.

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
