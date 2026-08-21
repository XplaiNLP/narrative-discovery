import argparse
import csv
import json
import os
from collections import Counter
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def krippendorff_alpha_nominal(units):
    pairs = []
    for u in units:
        u = [v for v in u if v is not None]
        if len(u) < 2:
            continue
        for a, b in combinations(u, 2):
            pairs.append((a, b))
    if not pairs:
        return None
    n_pairs = len(pairs)
    observed = sum(1 for a, b in pairs if a != b) / n_pairs
    flat = [v for u in units for v in u if v is not None]
    counts = Counter(flat)
    n = len(flat)
    expected = 1.0 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else 0.0
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def coerce(v):
    if v is None or v == "" or v == "nan":
        return None
    return v


def load_annotations(paths, key="label_match"):
    units = {}
    annotators = []
    for path in paths:
        ann = os.path.splitext(os.path.basename(path))[0]
        annotators.append(ann)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            uid = str(row["gs_id"])
            units.setdefault(uid, {})[ann] = coerce(row.get(key))
    matrix = []
    for uid, by_ann in units.items():
        matrix.append([by_ann.get(a) for a in annotators])
    return annotators, matrix


def load_votes(path, key="is_narr"):
    """Load the two-annotator combined votes file (per_annotator_votes.csv).

    Columns are ann_a_<key> / ann_b_<key> with key in {is_narr, label_match,
    confidence}. Returns (annotators, matrix) in the same shape as
    load_annotations, so the rest of the pipeline is unchanged.
    """
    if key == "is_disinfo_narrative":
        key = "is_narr"
    df = pd.read_csv(path)
    col_a, col_b = f"ann_a_{key}", f"ann_b_{key}"

    def cell(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        return None if s in ("", "nan") else s

    matrix = [[cell(r[col_a]), cell(r[col_b])] for _, r in df.iterrows()]
    return ["a", "b"], matrix


def rates(units, annotators):
    out = {}
    for ann in annotators:
        labels = [u[annotators.index(ann)] for u in units if u[annotators.index(ann)] is not None]
        n = len(labels)
        if n == 0:
            out[ann] = {}
            continue
        c = Counter(labels)
        out[ann] = {
            "n": n,
            "match_rate": sum(v for k, v in c.items() if k.startswith("cand_")) / n,
            "other_rate": c.get("Other", 0) / n,
            "none_rate": c.get("None", 0) / n,
        }
    return out


def votes_rates(df):
    """Per-annotator label rates for the votes file, from the canonical
    merged_a/merged_b columns (values in {match, other, none, no}). Using the
    normalised columns avoids matching the free-text label_match strings."""
    out = {}
    for ann in ("a", "b"):
        vals = [str(v).strip() for v in df[f"merged_{ann}"]
                if pd.notna(v) and str(v).strip()]
        n = len(vals)
        if n == 0:
            out[ann] = {}
            continue
        c = Counter(vals)
        out[ann] = {
            "n": n,
            "yes_rate": (c.get("match", 0) + c.get("other", 0) + c.get("none", 0)) / n,
            "match_rate": c.get("match", 0) / n,
            "other_rate": c.get("other", 0) / n,
            "none_rate": c.get("none", 0) / n,
        }
    return out


def votes_kappa(df, key):
    """Cohen's κ between annotators a and b on ann_a_<key>/ann_b_<key>.

    All rows are kept and a blank vote is treated as its own category (matching
    the paper's Table 12 convention), rather than pairwise-deleted."""
    if key == "is_disinfo_narrative":
        key = "is_narr"
    ca, cb = f"ann_a_{key}", f"ann_b_{key}"

    def lab(v):
        s = "" if pd.isna(v) else str(v).strip()
        return s or "(blank)"

    a = [lab(v) for v in df[ca]]
    b = [lab(v) for v in df[cb]]
    return round(cohen_kappa_score(a, b), 4) if a else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+",
                   help="Per-annotator CSVs from llm_judge.py or hand-annotated sheets")
    p.add_argument("--votes",
                   help="Combined two-annotator votes file (per_annotator_votes.csv)")
    p.add_argument("--key", default="label_match",
                   choices=["label_match", "is_disinfo_narrative", "is_narr", "confidence"])
    p.add_argument("--per-run", action="store_true",
                   help="votes mode: also report κ + rates per (dataset, pipeline) run, "
                        "matching the paper's per-run Table 12 (the default output pools all runs)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not args.input and not args.votes:
        p.error("provide either --votes <combined.csv> or --input <per-annotator.csv ...>")
    df_votes = None
    if args.votes:
        df_votes = pd.read_csv(args.votes)
        annotators, units = load_votes(args.votes, key=args.key)
    else:
        annotators, units = load_annotations(args.input, key=args.key)

    pair_kappa = {}
    for a, b in combinations(annotators, 2):
        ai, bi = annotators.index(a), annotators.index(b)
        a_lbls = [u[ai] for u in units if u[ai] is not None and u[bi] is not None]
        b_lbls = [u[bi] for u in units if u[ai] is not None and u[bi] is not None]
        if a_lbls:
            pair_kappa[f"{a}__{b}"] = round(cohen_kappa_score(a_lbls, b_lbls), 4)

    alpha = krippendorff_alpha_nominal(units)
    per_ann = votes_rates(df_votes) if args.votes else rates(units, annotators)

    out = {
        "key": args.key,
        "annotators": annotators,
        "n_units": len(units),
        "aggregation": "pooled across all runs" if not args.per_run else "pooled + per-run",
        "krippendorff_alpha_nominal": round(alpha, 4) if alpha is not None else None,
        "cohen_kappa_pairwise": pair_kappa,
        "rates_per_annotator": per_ann,
    }
    if args.per_run and df_votes is not None:
        out["per_run"] = {
            f"{ds}/{pipe}": {
                "n": len(g),
                f"kappa_{args.key}_a_b": votes_kappa(g, args.key),
                "rates_per_annotator": votes_rates(g),
            }
            for (ds, pipe), g in df_votes.groupby(["dataset", "pipeline"])
        }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
