# Appendix source attribution

Companion files for the appendix narrative tables. For every narrative shown
in the paper, this directory lists the source documents that contributed to
it, traceable down to the original text content.

**Scope: paper-tables only.** The two source-document files contain only the
texts/ads referenced by narratives that appear in the appendix tables — not
the full PN/CO corpora.

| corpus | full | in this directory |
| --- | ---: | ---: |
| PN (`datasets/pn/pn_all.csv`)  | 1892 texts | 1543 (82%) |
| CO (`datasets/co/all.csv`)     | 1330 ads   | 381 (29%) |

The full corpora live in their original locations (`datasets/pn/pn_all.csv`,
`datasets/co/all.csv`) and are unmodified.

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

- **`per_annotator_votes.csv`** — the raw **two-annotator** votes (annotators
  `a` and `b`) behind the human-validation numbers, one row per annotated
  candidate across the four persona runs (1041 rows). This is the source of
  truth for Table 3 (rates), Table 12 (Cohen's κ), and the strict/union
  candidate slices; the appendix tables above are the curated `other`/`none`
  outcomes derived from it.

  | column | description |
  | --- | --- |
  | `dataset` | `co` or `pn` |
  | `pipeline` | `cluster` (P_per-clr) or `graph` (P_per-com) |
  | `run_label` / `run_short` | identifiers of the pipeline run |
  | `item_id` | the narrative's id within the run (e.g., `hle_4`) |
  | `community_id` | underlying graph community / HDBSCAN cluster id |
  | `superclaim` | the narrative text shown to annotators |
  | `ann_a_is_narr` / `ann_b_is_narr` | raw `is_disinfo_narrative` vote (`yes`/`no`) |
  | `ann_a_label_match` / `ann_b_label_match` | raw label-match answer (a candidate label, `Other (in-domain, no match)`, or `None (out-of-domain)`) |
  | `ann_a_confidence` / `ann_b_confidence` | 1–5 self-rated confidence |
  | `merged_a` / `merged_b` | normalised label in `{match, other, none, no}` |

  Reproduce the paper numbers directly, e.g. the CO discovery counts
  (16 both-`Other` + 8 either-`None` = 24 candidates, 6 grouped as
  State Regulation & Free Market) or the inter-annotator agreement:

  ```bash
  # Cohen's κ (a vs b) + Krippendorff's α on the is-narrative decision.
  # Default pools all runs; --per-run gives the per-cell κ that matches Table 12.
  python ../iaa.py --votes per_annotator_votes.csv --key is_narr --per-run
  ```

- **`pn_source_documents.csv`** — every unique PN source text referenced.

  | column | description |
  | --- | --- |
  | `text_hash_id` | SHA-256 of the source text |
  | `language` | one of `BG`, `EN`, `HI`, `PT`, `RU`, `RU_2`, or empty |
  | `split` | `train`, `dev`, or empty |
  | `raw_filename` | name of the file in `datasets/pn/{split}/{language}/raw-documents/` whose SHA-256 byte-matches the canonical text; empty if the canonical text is a processed (whitespace-normalised, etc.) version of the raw file |
  | `text` | the canonical text content (from `datasets/pn/pn_all.csv`) |

- **`co_source_documents.csv`** — every unique CO ad referenced.

  | column | description |
  | --- | --- |
  | `ad_id` | matches the `id` column in `datasets/co/all.csv` |
  | `text_hash_id` | SHA-256 of the canonical ad-creative-body text |
  | `ad_creative_body` | the ad text as in `datasets/co/all.csv` |

## Joining

```python
import pandas as pd

narr = pd.read_csv("appendix_narratives.csv")
pn   = pd.read_csv("pn_source_documents.csv")
co   = pd.read_csv("co_source_documents.csv")

# All sources for the #1 graph-pipeline narrative in pn-other-graph
row = narr.query("table_label == 'pn-other-graph' and position == 1").iloc[0]
ids = row.source_doc_ids.split(";")
sources = (pn if row.dataset == "pn" else co).query(
    "text_hash_id in @ids" if row.dataset == "pn" else "ad_id in @ids"
)
```
