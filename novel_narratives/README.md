# Appendix source attribution

Companion files for the appendix narrative tables. For every narrative shown
in the paper, this directory lists the source documents that contributed to
it, by document id. The source texts themselves are **not** redistributed
here (see the paper's Ethics Statement); they can be looked up in the
original dataset releases with the ids and hashes below.

**Scope: paper-tables only.** The two source-document files list only the
documents referenced by narratives that appear in the appendix tables, not
the full PN/CO corpora.

| corpus | full | referenced in this directory |
| --- | ---: | ---: |
| PN (SemEval-2025 Task 10 release of PolyNarrative, train + dev, deduplicated by text) | 1892 texts | 1543 (82%) |
| CO (Climate Obstruction, `all.csv`) | 1330 ads | 381 (29%) |

The full corpora are obtained from their original releases (PolyNarrative via
SemEval-2025 Task 10; Climate Obstruction via `climate-nlp/climate-obstruction-narratives`)
and are used unmodified.

## Files

- **`appendix_narratives.csv`** — one row per narrative shown in the
  appendix tables.

  | column | description |
  | --- | --- |
  | `table_label` | which appendix table the row belongs to (`co-ood`, `co-other`, `pn-ood`, `pn-other-graph`, `pn-other-cluster`) |
  | `position` | 1-based row number within the table's pipeline block |
  | `dataset` | `co` (Climate Obstruction) or `pn` (PolyNarrative) |
  | `pipeline` | `graph` or `cluster` |
  | `run_label` / `run_short` | identifiers of the pipeline run |
  | `hle_id` | the narrative's id within the run (e.g., `hle_4`) |
  | `community_id` | the underlying graph community or HDBSCAN cluster id |
  | `superclaim` | the narrative text shown to annotators |
  | `is_singleton` | `1` if the underlying graph community has size 1 |
  | `n_sources` | number of unique source documents contributing claims |
  | `source_doc_ids` | `;`-joined list: `text_hash_id` (PN) or `ad_id` (CO) |

  Note: a row with `n_sources = 1` is not necessarily a singleton. Several
  claims extracted from the same document can form a community with more
  than one node; `is_singleton` is the authoritative flag.

- **`per_annotator_votes.csv`** — the raw **two-annotator** votes (annotators
  `a` and `b`) behind the human-validation numbers, one row per annotated
  narrative label across the four persona runs (1041 rows). This is the
  source of truth for Table 3 (rates), Table 4 (singletons per agreement
  group), Table 13 (inter-annotator agreement) and the strict/union
  candidate groups; the appendix tables above are the curated `other`/`none`
  outcomes derived from it.

  | column | description |
  | --- | --- |
  | `dataset` | `co` or `pn` |
  | `pipeline` | `cluster` (P_per-clr) or `graph` (P_per-com) |
  | `run_label` / `run_short` | identifiers of the pipeline run |
  | `item_id` | the narrative's id within the run (e.g., `hle_4`) |
  | `community_id` | underlying graph community / HDBSCAN cluster id |
  | `superclaim` | the narrative text shown to annotators |
  | `ann_a_is_narr` / `ann_b_is_narr` | raw `is_disinfo_narrative` vote (`yes` / `no` / `unclear`; `unclear` was an allowed answer, see the guidelines in the paper's appendix) |
  | `ann_a_label_match` / `ann_b_label_match` | raw label-match answer (a candidate label, `Other (in-domain, no match)`, or `None (out-of-domain)`) |
  | `ann_a_confidence` / `ann_b_confidence` | 1–5 self-rated confidence |
  | `merged_a` / `merged_b` | normalised label in `{match, other, none, no}`; `no` covers `no` and `unclear` |

  Counting rule (as in the guidelines): a `label_match` answer counts only
  when the same annotator's `is_disinfo_narrative` is `yes`. The raw columns
  are kept exactly as annotated; the rule is applied in the `merged_*`
  columns. Two rows are affected: `hle_391` (PN, graph), where annotator `b`
  answered `unclear` and still filled `None (out-of-domain)`, so `merged_b`
  is `no`; and `hle_126` (PN, cluster), where annotator `a` answered `yes`
  but left `label_match` blank, so the row counts in the yes-rate only and
  `merged_a` is `no`. One further row, `hle_173` (CO, graph), has a blank
  `ann_b_is_narr`; it counts as not-yes (`merged_b = no`). With this rule the
  file reproduces Tables 3, 4, 13 and 18–22 of the paper exactly; κ and
  agreement on `label_match` are computed on the raw answer strings with a
  blank answer as its own category (`iaa.py`).

  Reproduce the paper numbers directly, e.g. the CO discovery counts
  (16 both-`Other` + 8 either-`None` = 24 candidates, 6 grouped as
  State Regulation & Free Market) or the inter-annotator agreement:

  ```bash
  # Cohen's κ (a vs b) + Krippendorff's α on the is-narrative decision.
  # Default pools all runs; --per-run gives the per-cell κ that matches Table 13.
  python ../iaa.py --votes per_annotator_votes.csv --key is_narr --per-run
  ```

- **`pn_source_documents.csv`** — every unique PN source document referenced,
  by id.

  | column | description |
  | --- | --- |
  | `text_hash_id` | SHA-256 of the canonical (whitespace-normalised) text; use it to verify a document retrieved from the original release |
  | `language` | one of `BG`, `EN`, `HI`, `PT`, `RU`, `RU_2`, or empty |
  | `split` | `train`, `dev`, or empty |
  | `raw_filename` | name of the file in the SemEval-2025 Task 10 release (`{split}/{language}/raw-documents/`) whose SHA-256 byte-matches the canonical text; empty if the canonical text is a processed version of the raw file |

- **`co_source_documents.csv`** — every unique CO ad referenced, by id.

  | column | description |
  | --- | --- |
  | `ad_id` | matches the `id` column in the Climate Obstruction release (`all.csv`) |
  | `text_hash_id` | SHA-256 of the canonical ad-creative-body text |

## Joining

```python
import pandas as pd

narr = pd.read_csv("appendix_narratives.csv")
pn   = pd.read_csv("pn_source_documents.csv")
co   = pd.read_csv("co_source_documents.csv")

# All source ids for the #1 graph-pipeline narrative in pn-other-graph
row = narr.query("table_label == 'pn-other-graph' and position == 1").iloc[0]
ids = row.source_doc_ids.split(";")
sources = (pn if row.dataset == "pn" else co).query(
    "text_hash_id in @ids" if row.dataset == "pn" else "ad_id in @ids"
)
# Look the documents up in the original release via raw_filename (PN) or ad_id (CO).
```
