# Narrative-mining pipelines

Six end-to-end pipelines for extracting narrative-level "super-claims" from a
corpus of short texts (ads, tweets, news snippets, propaganda, …), plus the
evaluation, human-validation, and topical-coverage tooling used in the paper.

> **Data and content notice.** The released files in `novel_narratives/` are
> narrative candidates generated from disinformation corpora, plus the raw votes
> of the two annotators who validated them. They are **not gold labels**. Source
> documents are referenced by id only and are not redistributed. Code is MIT,
> the released data is CC BY 4.0, and the label sets in `taxonomies/` stay under
> the terms of their source datasets. See [TERMS_OF_USE.md](TERMS_OF_USE.md).

| `--method`   | Stages                                                                          |
|--------------|---------------------------------------------------------------------------------|
| `P_SVO`      | spaCy SVO triplets → embed → UMAP → HDBSCAN → centroid triplet (no LLM)          |
| `P_dinam`    | LLM claim extraction → embed → UMAP → HDBSCAN → DiNaM 15-word narrative         |
| `P_clr`      | LLM claim extraction → embed → UMAP → HDBSCAN → super-claim                     |
| `P_com`      | LLM text decomposition → graph + attributes → Leiden → super-claim              |
| `P_per-clr`  | LLM persona claim extraction (4 personas) → embed → UMAP → HDBSCAN → super-claim |
| `P_per-com`  | Persona claims → text decomposition → graph + attributes → Leiden + back-join → super-claim |


All pipelines write `group_summaries.csv` (`human_readable_id, title, description, hash_id, community_id`) and `result.json` into the run directory.

## Installation

```bash
conda create -n narrative python=3.10 -y
conda activate narrative

pip install pandas numpy scipy scikit-learn pyyaml networkx \
            pydantic openai httpx backoff \
            sentence-transformers umap-learn hdbscan \
            python-igraph leidenalg \
            spacy
python -m spacy download en_core_web_lg
```

For `P_SVO`/`P_clr`/`P_per-clr`/`P_dinam` you need an OpenAI-compatible
endpoint. The pipelines were developed against vLLM serving Gemma-4-31B-it/Qwen
locally; OpenAI itself works too. Configure it in `config.yaml` under `llm:`.

```bash
# vLLM example
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B-it \
    --port 11434 --api-key your-secret-key
```

## Quick start — Climate Obstruction (CO)

The CO corpus is published at
<https://github.com/climate-nlp/climate-obstruction-narratives/tree/main/Data>.
The other datasets used in the paper are not redistributed here; the same
recipe applies once you have a CSV with one text per row.

```bash
# 1. Download CO's all.csv into ./datasets/co/all.csv
mkdir -p datasets/co
curl -L -o datasets/co/all.csv \
    https://raw.githubusercontent.com/climate-nlp/climate-obstruction-narratives/main/Data/all.csv

# 2. Convert to the canonical input format (text_hash_id, text).
#    CO stores the ad copy in ad_creative_body, hence --text-col.
python prepare_input.py datasets/co/all.csv runs/co/input_texts.csv \
    --text-col ad_creative_body

# 3. Run a pipeline. Method id is one of P_SVO, P_clr, P_per-clr, P_com,
#    P_per-com, P_dinam. Settings (LLM, embed model, UMAP/HDBSCAN/Leiden) are
#    in config.yaml.
python main.py \
    --method P_clr \
    --input-texts runs/co/input_texts.csv \
    --output-dir runs \
    --run-label co_P_clr

# 4. Evaluate against the published CO taxonomy. Hungarian/WCD/collapse.
python eval.py \
    --run-dir runs/co_P_clr \
    --taxonomy taxonomies/co.json
```

The taxonomy JSONs in `taxonomies/` are flat `{ref_id: text}` dictionaries.
Add a new dataset by dropping its taxonomy in there. `taxonomies/eu.json` has
one entry per distinct (Topic, Broad_narrative) pair of the EU DisinfoTest
release, formatted `Topic: Broad_narrative`, 170 in total (the release has 167
distinct broad-narrative strings; two of them occur under several topics).

## Inputs

`main.py` reads a CSV with at least a `text` column. If `text_hash_id` is also
present it is reused; otherwise it is derived as `sha256(text)`. Use
`prepare_input.py` to produce that file from any source CSV.

## Configuration

All knobs live in `config.yaml`. Override individual fields on the command
line with `--method`, `--input-texts`, `--output-dir`, `--run-label`. The
sections worth knowing:

- `llm`: provider (`vllm` or `openai`), model id, endpoint, concurrency.
- `embed`: SentenceTransformer model + device for the pipeline embedding step.
  The paper uses `Qwen/Qwen3-Embedding-4B` on `cuda`. (The closed-world eval
  metrics use a *different* encoder, `microsoft/harrier-oss-v1-0.6b` — see
  Evaluation; do not confuse the two.)
- `umap`, `hdbscan`: clustering parameters (matched to DiNaM defaults).
- `leiden`: `partition_type` is `modularity`, `cpm`, or `mod_rb`.
- `extract.personas`: persona keys (subset of `journalist`,
  `political_scientist`, `conspiracy_analyst`, `fact_checker`).

## Outputs

A successful run writes into `<output_dir>/<run_label>/`:

- `group_summaries.csv` — final HLE-equivalent table (one row per super-claim or DiNaM narrative).
- `result.json` — `{label, method, n_summaries, topic_descriptions}`.
- `community_summaries.jsonl` — per-cluster raw LLM responses.
- Per-stage artifacts: `all_claims.csv` / `svo_triplets.csv` /
  `text_decomposition.jsonl` / `entities.csv` / `semantic_units.csv` /
  `relationships.csv` / `attributes.csv` / `graph.pkl` /
  `community_assignments.csv`, depending on the method.

## Evaluation

`eval.py` embeds both the run's group summaries and a taxonomy with the
Harrier embedding model and reports:

- `hungarian` — mean cosine similarity under a Hungarian assignment.
- `wcd` — Weighted Chamfer Distance (lower is better).
- `collapse` and `c_over_r` — number of reference labels that collapse onto the
  same group summary under the top-1 rule, and that count divided by the number
  of reference labels.

```bash
python eval.py --run-dir runs/co_P_clr --taxonomy taxonomies/co.json
```

## Human validation

Three scripts cover the human/LLM-judge path used in the paper:

```bash
# 1. Build a candidate sheet for a run: top-2 nearest entries PER taxonomy
#    (own + reference), deterministically shuffled. CO's reference is CARDS2;
#    PN's reference is EU DisinfoTest. Pass any number of taxonomies as label=path.
python human_val.py \
    --run-dir runs/co_P_clr \
    --taxonomies own=taxonomies/co.json ref=taxonomies/cards2.json \
    --per-taxonomy 2 --max-rows 100 \
    --out runs/co_P_clr/human_sheet.csv

# 2. Run the LLM-judge panel over the same sheet: the four pipeline personas on
#    two backbone models (Gemma-4-31B-it + Qwen3.5-27B) = 8 judges, majority vote
#    (appendix H.2). Point --api-bases at the vLLM endpoint(s) serving each model.
python llm_judge.py \
    --input runs/co_P_clr/human_sheet.csv \
    --output runs/co_P_clr/llm_judged.csv

# 3. Inter-annotator agreement (Krippendorff's α + pairwise Cohen's κ).
#    Runs on the released two-annotator votes file (annotators a, b), or on any
#    set of per-annotator CSVs that share the human-sheet schema. The default
#    output POOLS all four runs; add --per-run for the per-(dataset,pipeline)
#    κ that reproduces the paper's Table 12.
python iaa.py --votes novel_narratives/per_annotator_votes.csv --key is_narr --per-run
```

## Topical coverage

Zero-shot 4-way topic classification of group summaries (Ukraine / Climate /
Other / Noise) with prompt-order ablation and self-consistency voting:

```bash
python topical_coverage.py --run-dir runs/co_P_clr --variants A B C --k 5
```

Writes `topic_per_item.csv` and `topic_summary.json` next to the run.

## License and citation

| Material | Terms |
| --- | --- |
| Code (`*.py`, `config.yaml`) | [MIT](LICENSE) |
| Released data (`novel_narratives/`) | [CC BY 4.0](LICENSE-DATA) |
| Label sets (`taxonomies/`) | terms of the source datasets, see [TERMS_OF_USE.md](TERMS_OF_USE.md) |

Full terms, source-dataset attribution and the contact for corrections and
removal requests: [TERMS_OF_USE.md](TERMS_OF_USE.md).

If you use this material, please cite:

> Max Upravitelev, Veronika Solopova, Jing Yang, Charlott Jakob, Alexandra
> Tsiakalou, Neda Foroutan and Vera Schmitt. 2026. From Repetition to
> Recognition: Inductive Discovery of Disinformation Narratives. In *Findings of
> the Association for Computational Linguistics: EMNLP 2026*.
