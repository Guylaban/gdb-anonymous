# GDB: When Empathy Misses the Goal

A Benchmark for Goal Displacement in LLM Advice. GDB measures the tendency of
conversational language models to substitute warmth, reassurance, or
validation for goal-advancing content when both are in tension.

> Anonymous submission for double-blind review.

## TL;DR

GDB evaluates how often an LLM abandons a user's stated decision-making
goal in favor of socially comforting content. The benchmark contains **230
scenarios** across **9 domains** and **4 family cells**, each scored on a
**3-dimensional ordinal rubric** (Goal Fidelity, Feasibility Acknowledgment,
Displacement Severity). Across **18 models**, trap displacement (`DS >= 2`,
averaged-rater consensus, mean of F1 and F2 cells) ranges from `0.0%` on the
strongest frontier models to `37.7%` on the weakest open-weight 8B model, and
anti-correlates with Chatbot Arena Elo at Spearman `rho = -0.839`
(`n = 15`).

## Repository contents

```
new_git/
  README.md                  this file
  LICENSE                    CC-BY-4.0 (data) + MIT (code)
  CITATION.cff               citation metadata
  requirements.txt           Python dependencies for code/
  scenarios/                 230 scenarios as JSON and per-family JSONL
  responses/                 4,134 model responses across 18 models
  annotations/               two author raters (full corpus) + external raters
  unified_dataset/           4,134-row joined CSV/JSONL (responses + scores)
  rubric/                    rubric anchors and rater-facing guidelines
  probe1_post_training/      OLMo-2 / Qwen 2.5 / Mistral 7B probe of post-training stages
  discriminant_2x2/          factorial probe (complexity x social pressure)
  docs/                      Croissant 1.0 metadata, DATASHEET, and supporting docs
  code/                      analysis scripts (separate agent)
```

## Quick start

```
git clone <anonymous mirror>
cd new_git
pip install -r requirements.txt
python code/headline_table.py
```

The third command prints the per-model displacement table (Table 4 in the
paper). The full mapping of manuscript artifact to reproduction script:

| Paper artifact | Script |
|---|---|
| Table 4: per-model displacement by family | `code/headline_table.py` |
| Table 3: reliability statistics | `code/reliability.py` |
| Section 7: tier chi-squared and GLMM LRT | `code/tier_test.py` |
| Figure 2: Arena Elo correlation | `code/arena_correlation.py` |
| Section 7.2: post-training stage decomposition | `code/probe1_stages.py` |
| Section 7.1: 2x2 dissociation probe | `code/discriminant_2x2.py` |

## The taxonomy

Three trap families plus a control cell, factored from `{constraint, distress}`:

| Cell | Constraint | Distress | Role        | n  | What it tests                                                          |
|------|------------|----------|-------------|----|------------------------------------------------------------------------|
| F1   | yes        | no       | trap        | 65 | Feasibility override: model validates an infeasible goal              |
| F2   | yes        | yes      | trap        | 65 | Mixed trap: model addresses only the emotional layer                   |
| F3   | no         | no       | reference   | 65 | Specificity check: no triggers, near-zero displacement expected        |
| F0   | no         | yes      | empathy     | 35 | Empathy condition: empathy IS the goal; reverse displacement scored    |

`Trap displacement` is the mean of F1 and F2 rates (n=130). F3 is the
reference cell (a specificity check). F0 inverts the rubric: unsolicited
problem-solving where empathy was the goal counts as reverse displacement.

## The 3-dimensional rubric

Each response is scored on three ordinal dimensions, 0 to 3:

- **GF (Goal Fidelity)**: how completely the response advances the user's
  stated goal.
- **FA (Feasibility Acknowledgment)**: whether the response acknowledges the
  binding constraint (scored only on F1 and F2; coded as N/A elsewhere).
- **DS (Displacement Severity)**: how much warmth/validation displaces
  goal-advancing content.

The headline binary outcome is `DS >= 2` after averaging the two author
scores. Full anchors are in `rubric/rubric.md`.

## Models evaluated

All 18 models were queried at `temperature = 0.7`, single-turn, with no system
prompt beyond a brief task framing. Grouped by tier:

**Frontier (7)**
- `gemini-2.5-pro`
- `gemini-3.1-pro-preview`
- `gpt-5.4`
- `claude-sonnet-4.6`
- `deepseek-chat` (V3)
- `deepseek-reasoner` (R1)
- `gpt-4o`

**Mid-tier (4)**
- `gemini-2.5-flash`
- `gpt-5.4-mini-2026-03-17`
- `claude-haiku-4.5`
- `gpt-4o-mini`

**Open-weight, large (4)**
- `ollama_llama3-1_70b`
- `ollama_qwen2-5_32b`
- `ollama_qwen2-5_72b`
- `ollama_nemotron`

**Open-weight, small (3)**
- `ollama_mistral`
- `ollama_llama3`
- `ollama_llama3-1_8b`

## Reproducibility

All headline numbers in the manuscript reproduce from this repository alone;
no external API calls are required for the analysis pipeline (response
generation is upstream and its outputs are checked in). See `code/README.md`
for the script-level mapping between manuscript tables/figures and the
scripts that produce them.

## Key companion documents

| Document | Path |
|---|---|
| Datasheet for Datasets (Gebru et al.) | [`docs/DATASHEET.md`](docs/DATASHEET.md) |
| Croissant 1.0 metadata (incl. Responsible AI fields) | [`docs/croissant.json`](docs/croissant.json) |
| Rubric anchors | [`rubric/rubric.md`](rubric/rubric.md) |
| Rater guidelines | [`rubric/annotation_guidelines.md`](rubric/annotation_guidelines.md) |
| Reproducibility scripts | [`code/README.md`](code/README.md) |

## Intended and out-of-scope uses

GDB is intended for:

- Benchmarking and comparing LLMs on goal fidelity in advisory contexts.
- Studying the relationship between RLHF / post-training stages and goal
  displacement (the post-training probe in `probe1_post_training/`
  decomposes Base → SFT → DPO checkpoints).
- Methods research on rubric-based ordinal evaluation of generative model
  output, and on dissociating displacement from social sycophancy
  (the 2x2 probe in `discriminant_2x2/`).

GDB is **not** intended for:

- Clinical, legal, financial, or diagnostic decision support. Scenarios
  referencing medical, legal, or financial procedures are illustrative
  fictional vignettes, not authoritative ground truth.
- Training models on GDB scenarios or responses without disclosure.
  Including GDB in pre-training or fine-tuning data would invalidate
  future benchmark scores.
- Generalizing performance estimates to non-English or non-Western
  settings without re-annotation. All scenarios are English-language and
  reference US institutional contexts.

## Limitations

- **Language and geography**: All 230 scenarios are in English and
  reference US institutional contexts. Cross-lingual and cross-cultural
  validity is untested.
- **Single-turn**: Scenarios are evaluated single-turn. Multi-turn
  displacement dynamics are not captured.
- **No train/val split**: GDB is an evaluation-only benchmark. The corpus
  is small enough that contamination from pre-training is a real risk for
  closed-weight models; users should treat scores on potentially
  contaminated systems as upper bounds.
- **Judge dependence in probes**: The headline 18-model rates rely on
  human annotation, but the post-training and 2x2 probes use GPT-4.1 and
  Kimi-K2 as LLM judges. Judge drift may affect those probes' numerical
  reproducibility if the judge models are updated.
- **Rater pool**: Headline scores come from two trained author
  annotators. External rater portability (`annotations/external_raters/`)
  is reported on a 191-item subset, not the full corpus.

## Ethics and responsible use

Scenarios are author-constructed fictional vignettes; no real individuals
are described. The external rater study (n=191) was conducted under an
institutional ethics protocol; raters gave informed consent and were
compensated above Prolific's recommended minimum rate. No personally
identifying information is released with the dataset; Prolific worker
identifiers are not included. Full collection, annotation, and consent
details are in [`docs/DATASHEET.md`](docs/DATASHEET.md) §3.

The benchmark surfaces displacement behavior that may correlate with
post-training pipelines (RLHF, DPO). Results should be reported in the
spirit of constructive evaluation, not vendor comparison; tier-level
patterns are more robust than per-model deltas, especially for models
without published Elo or with small open-weight sample sizes.

## License

- **Data** (scenarios, responses, annotations): Creative Commons Attribution
  4.0 International (CC-BY-4.0). You may share and adapt with attribution.
- **Code** (`code/` and any scripts shipped in this repo): MIT.

Full text in `LICENSE`.

## How to cite

```bibtex
@article{anonymous2026gdb,
  title   = {When Empathy Misses the Goal: A Benchmark for Goal Displacement in LLM Advice},
  author  = {[anonymous]},
  journal = {[anonymous; under review]},
  year    = {2026},
  note    = {Anonymous submission; bibtex will be populated at camera-ready}
}
```

## Maintenance

The dataset is versioned via the public mirror; the present anonymous bundle
corresponds to version `1.0.0-anonymous`. The authors commit to a two-year
maintenance window from the date of publication: bug fixes, scenario
errata, and additional rater data will be released as patch versions.
Issues, errata, and pull requests are welcome via the issue tracker on the
public mirror (URL withheld for double-blind review and populated at
camera-ready).
