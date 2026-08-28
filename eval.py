import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer


HARRIER = "microsoft/harrier-oss-v1-0.6b"
HARRIER_QUERY_PROMPT = "sts_query"


def load_taxonomy(path):
    with open(path) as f:
        return json.load(f)


def load_run(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "group_summaries.csv"))
    return df["human_readable_id"].astype(str).tolist(), df["description"].astype(str).tolist()


def embed(model, texts, as_query=False):
    kwargs = {"batch_size": 64, "normalize_embeddings": True,
              "show_progress_bar": len(texts) > 200}
    if as_query and "harrier" in model[0].auto_model.config.name_or_path.lower():
        kwargs["prompt_name"] = HARRIER_QUERY_PROMPT
    return np.asarray(model.encode(texts, **kwargs), dtype=np.float32)


def metrics(ref_ids, ref_emb, gs_ids, gs_emb):
    sim = ref_emb @ gs_emb.T
    dist = 1.0 - sim
    n_ref, n_gs = sim.shape

    coverage = float(dist.min(axis=1).mean())
    precision = float(dist.min(axis=0).mean())
    wcd = (n_ref / (n_ref + n_gs)) * coverage + (n_gs / (n_ref + n_gs)) * precision

    row, col = linear_sum_assignment(dist)
    hungarian = float(np.mean([sim[r, c] for r, c in zip(row, col)]))

    top1 = sim.argmax(axis=1)
    demand = defaultdict(list)
    for i, j in enumerate(top1):
        demand[int(j)].append(ref_ids[i])
    collapse = sum(len(v) - 1 for v in demand.values() if len(v) > 1)

    return {
        "n_summaries": n_gs,
        "n_ref": n_ref,
        "hungarian": round(hungarian, 4),
        "wcd": round(wcd, 4),
        "collapse": collapse,
        "c_over_r": round(collapse / n_ref, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Pipeline run directory containing group_summaries.csv")
    p.add_argument("--taxonomy", required=True, help="JSON file mapping ref_id → text")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tax = load_taxonomy(args.taxonomy)
    ref_ids = sorted(tax.keys())
    ref_texts = [tax[k] for k in ref_ids]
    gs_ids, gs_texts = load_run(args.run_dir)

    model = SentenceTransformer(HARRIER, device=args.device, trust_remote_code=True)
    ref_emb = np.asarray(model.encode(ref_texts, batch_size=64, normalize_embeddings=True,
                                      show_progress_bar=False, prompt_name=HARRIER_QUERY_PROMPT),
                         dtype=np.float32)
    gs_emb = np.asarray(model.encode(gs_texts, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=len(gs_texts) > 500),
                        dtype=np.float32)

    result = metrics(ref_ids, ref_emb, gs_ids, gs_emb)
    print(json.dumps(result, indent=2))
    out = os.path.join(args.run_dir, f"eval_{os.path.splitext(os.path.basename(args.taxonomy))[0]}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
