import argparse
import csv
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"


def deterministic_shuffle(items, seed_str):
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx], idx


def load_taxonomy(label, path):
    with open(path) as f:
        tax = json.load(f)
    return [{"id": tid, "text": text, "taxonomy": label} for tid, text in tax.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--taxonomies", nargs="+", required=True,
                   help="Pairs label=path, e.g. own=taxonomies/co.json ref=taxonomies/cards2.json")
    p.add_argument("--per-taxonomy", type=int, default=2,
                   help="top-N candidates retrieved from EACH taxonomy (paper: 2)")
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help="retrieval encoder; paper §3.1.2 uses Qwen3-4B-Embedding")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="Output CSV path")
    args = p.parse_args()

    taxonomies = dict(t.split("=", 1) for t in args.taxonomies)
    # one candidate sub-pool per taxonomy (retrieval is stratified per taxonomy)
    pools = {label: load_taxonomy(label, path) for label, path in taxonomies.items()}

    df = pd.read_csv(os.path.join(args.run_dir, "group_summaries.csv"))
    if args.max_rows and len(df) > args.max_rows:
        rng = random.Random(hashlib.sha256(args.run_dir.encode()).hexdigest()[:8])
        df = df.iloc[sorted(rng.sample(range(len(df)), args.max_rows))].reset_index(drop=True)

    model = SentenceTransformer(args.embed_model, device=args.device, trust_remote_code=True)
    gs_emb = np.asarray(model.encode(df["description"].tolist(), batch_size=64,
                                     normalize_embeddings=True, show_progress_bar=False),
                        dtype=np.float32)
    # per-taxonomy similarity matrices
    pool_sims = {}
    for label, pool in pools.items():
        pool_emb = np.asarray(model.encode([c["text"] for c in pool], batch_size=64,
                                           normalize_embeddings=True,
                                           show_progress_bar=len(pool) > 200),
                              dtype=np.float32)
        pool_sims[label] = gs_emb @ pool_emb.T

    total_k = args.per_taxonomy * len(pools)
    rows = []
    for i, gs_row in df.iterrows():
        cands = []
        for label, pool in pools.items():
            sims_i = pool_sims[label][i]
            top_idx = np.argsort(sims_i)[::-1][:args.per_taxonomy]
            cands += [(pool[j], float(sims_i[j])) for j in top_idx]
        shuffled, order = deterministic_shuffle(cands, f"{args.run_dir}|{gs_row['human_readable_id']}")
        out = {"gs_id": gs_row["human_readable_id"], "gs_description": gs_row["description"]}
        for k, (cand, sim) in enumerate(shuffled, 1):
            out[f"cand_{k}_id"] = cand["id"]
            out[f"cand_{k}_text"] = cand["text"]
            out[f"cand_{k}_taxonomy"] = cand["taxonomy"]
            out[f"cand_{k}_sim"] = f"{sim:.4f}"
        out["cand_order"] = json.dumps(order)
        out["is_disinfo_narrative"] = ""
        out["label_match"] = ""
        out["confidence"] = ""
        out["notes"] = ""
        rows.append(out)

    fieldnames = ["gs_id", "gs_description"]
    for k in range(1, total_k + 1):
        fieldnames += [f"cand_{k}_id", f"cand_{k}_text", f"cand_{k}_taxonomy", f"cand_{k}_sim"]
    fieldnames += ["cand_order", "is_disinfo_narrative", "label_match", "confidence", "notes"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows × {len(fieldnames)} cols → {args.out}")


if __name__ == "__main__":
    main()
