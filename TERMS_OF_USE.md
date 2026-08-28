# Terms of use

Companion release for **"From Repetition to Recognition: Inductive Discovery of
Disinformation Narratives"** (Findings of EMNLP 2026).

## 1. What is in this repository, and under which terms

| Material | Files | Terms |
| --- | --- | --- |
| Code | `*.py`, `config.yaml` | MIT, see [`LICENSE`](LICENSE) |
| Research data we produced | `novel_narratives/` | CC BY 4.0, see [`LICENSE-DATA`](LICENSE-DATA) |
| Taxonomy label sets | `taxonomies/` | Third-party. The terms of the original dataset apply, see §6 |
| Documentation | `README.md`, `novel_narratives/README.md`, this file | CC BY 4.0 |

Source documents are **not** in this repository. See §4.

## 2. How to cite

If you use anything from this repository, cite the paper. The citation is in
the [README](README.md#license-and-citation).

If you use a taxonomy file from `taxonomies/`, also cite the dataset it comes
from (§6). If you retrieve source documents through the ids in
`novel_narratives/`, cite the dataset that publishes them.

## 3. What the released narrative candidates are

`novel_narratives/` holds narrative labels that unsupervised pipelines generated
from the Climate Obstruction and PolyNarrative corpora, together with the raw
votes of the two annotators who validated them. They are released as discussion
material for the maintenance and extension of disinformation narrative
taxonomies, and so that the numbers in the paper can be recomputed.

They are **not gold labels**. The validation is a two-annotator pilot carried
out by two co-authors of the paper, and the paper documents substantial
disagreement between them: the two annotators marked largely disjoint sets of
labels as out-of-domain, with an overlap of 9/31, 14/53 and 1/8 across the three
runs where the label was used at all. No domain-expert panel has reviewed these
candidates, and none of them has entered any taxonomy. Read §3.1.2, §4, §5.3 and
Appendix J of the paper, and the counting rule in
[`novel_narratives/README.md`](novel_narratives/README.md), before using any of
this as training or evaluation data.

We would be glad to see these candidates used, checked and improved, and we are
interested in hearing from anyone who does: <max.upravitelev@tu-berlin.de>. One
request: please do not use this material to generate, optimize or amplify
disinformation. The candidates are formulations of misleading claims, and they
are published so that such claims can be recognized.

## 4. Source documents are not redistributed

`co_source_documents.csv` and `pn_source_documents.csv` carry document ids and
SHA-256 hashes only. No source text is included anywhere in this repository.

To obtain the texts, go to the original releases (§6), accept their terms, and
join on the ids. The hashes let you check that the document you retrieved is the
one a narrative label was derived from. `raw_filename` names the file in the
SemEval-2025 Task 10 release for PolyNarrative documents.

## 5. Content

The corpora and the generated labels are disinformation material: claims about
climate science, the war in Ukraine, COVID-19 and elections that are false or
misleading, along with the annotators' judgments about them. They refer to
public figures and to institutions. They contain no private individuals' data,
no identifiers and no contact information.

## 6. Source datasets

`taxonomies/` reproduces the label sets of the datasets below so that the
evaluation in the paper can be rerun. Each file stays under the terms of its
source dataset, which the table names where they are stated. Where a dataset
states no license, we reproduce its label set as attributed quotation for
research use; see §7.

| File | Dataset | Source | Terms |
| --- | --- | --- | --- |
| `ca.json` | CARDS | Coan et al. (2021), *Sci. Rep.*, [doi:10.1038/s41598-021-01714-4](https://doi.org/10.1038/s41598-021-01714-4) | see source |
| `cards2.json` | CARDS 2 | Coan et al. (2026), *Commun. Sustain.*, [doi:10.1038/s44458-025-00029-z](https://doi.org/10.1038/s44458-025-00029-z) | see source |
| `co.json` | Climate Obstruction | Rowlands et al. (2024), Findings of ACL; [climate-nlp/climate-obstruction-narratives](https://github.com/climate-nlp/climate-obstruction-narratives) | Apache-2.0 |
| `cv.json` | COVID conspiracy narratives (German Telegram) | Heinrich et al. (2024), LREC-COLING, [2024.lrec-main.173](https://aclanthology.org/2024.lrec-main.173/) | see source |
| `eu.json` | EU DisinfoTest | Sosnowski et al. (2024), Findings of EMNLP, [2024.findings-emnlp.862](https://aclanthology.org/2024.findings-emnlp.862/) | see source |
| `hp.json` | HALT-PROP | Rizgelienė et al. (2026), *Sci. Data*; [VilniausUniversitetas/HALT-PROP](https://huggingface.co/datasets/VilniausUniversitetas/HALT-PROP) | CC BY 4.0 |
| `nm.json` | Narrative Media Framing | Otmakhova and Frermann (2025), Findings of ACL, [2025.findings-acl.477](https://aclanthology.org/2025.findings-acl.477/) | see source |
| `pn.json` | PolyNarrative | Nikolaidis et al. (2025), ACL, [2025.acl-long.1513](https://aclanthology.org/2025.acl-long.1513/); release: Piskorski et al. (2025), SemEval-2025 Task 10 | see source |
| `uk.json` | UKElectionNarratives | Haouari et al. (2025), ICWSM, [doi:10.1609/icwsm.v19i1.35950](https://doi.org/10.1609/icwsm.v19i1.35950); [GateNLP/UKElectionNarratives](https://github.com/GateNLP/UKElectionNarratives) | see source |

The label strings are reproduced as published, with light normalization for the
evaluation code: each entry is one string that concatenates a label with its
description where the source separates them. `eu.json` has 170 entries, the
distinct (Topic, Broad_narrative) pairs of the EU DisinfoTest release.

## 7. Corrections and removal

If you hold rights in any material here, or you are an author of one of the
datasets in §6 and want an attribution corrected or a file removed, write to
<max.upravitelev@tu-berlin.de> and we will act on it.

The same address takes corrections to the data. The paper and
[`novel_narratives/README.md`](novel_narratives/README.md) record the known
irregularities in the annotation file, which are three votes that the counting
rule treats as not-a-yes.

## 8. No warranty

This material is provided as is, with no warranty of any kind, and with no
claim that the narrative candidates are accurate, complete or fit for any
purpose. The paper's Limitations section states what we know about their limits.
