import argparse
import asyncio
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from cluster import (
    build_graph,
    centroid_per_cluster,
    community_content_backjoin,
    community_content_raw,
    embed_and_cluster,
    gen_attributes,
    group_by_label,
    leiden_communities,
)
from extract import (
    claim_extract,
    persona_claim_extract,
    svo_extract,
    text_decompose,
)
from io_utils import (
    load_input_texts,
    synthesize_dinam_narrative,
    synthesize_superclaim,
)
from llm import LLMClient


METHOD_LABELS = {
    "P_SVO":     "harm_srl_superclaim",
    "P_clr":     "harm_dinam_superclaim",
    "P_per-clr": "harm_persona_superclaim",
    "P_com":     "harm_og_superclaim",
    "P_per-com": "harm_persona_og_superclaim",
    "P_dinam":   "Y_dinam",
}

STAGES = {
    "P_SVO":     ["svo", "cluster", "synthesize"],
    "P_clr":     ["extract", "cluster", "synthesize"],
    "P_dinam":   ["extract", "cluster", "synthesize"],
    "P_per-clr": ["extract", "cluster", "synthesize"],
    "P_com":     ["decompose", "graph", "attributes", "leiden", "synthesize"],
    "P_per-com": ["extract", "decompose", "graph", "attributes", "leiden", "synthesize"],
}


def _load_communities(out):
    df = pd.read_csv(os.path.join(out, "community_assignments.csv"))
    cd = defaultdict(list)
    for _, r in df.iterrows():
        cd[int(r["community_id"])].append(r["node_hash_id"])
    return [(cid, members) for cid, members in sorted(cd.items())]


async def run(cfg, llm_factory=LLMClient):
    method = cfg["method"]
    out = os.path.join(cfg["output_dir"], cfg["run_label"])
    os.makedirs(out, exist_ok=True)
    skip_until = cfg.get("skip_until")
    stop_after = cfg.get("stop_after")

    stages = STAGES[method]
    start_idx = stages.index(skip_until) if skip_until else 0

    def stop(stage):
        return stage == stop_after

    texts = load_input_texts(cfg["input_texts"])
    print(f"[{method}] {len(texts)} input texts → {out}")
    print(f"  stages: {stages}  start={stages[start_idx]}  stop_after={stop_after or 'end'}")

    llm = llm_factory(**cfg["llm"])

    df = decomp = G = communities = groups = None
    t0 = time.time()

    if "svo" in stages:
        if start_idx <= stages.index("svo"):
            df = svo_extract(texts, out, cfg["extract"]["spacy_model"],
                             cfg["extract"]["max_triplets_per_text"])
        else:
            df = pd.read_csv(os.path.join(out, "svo_triplets.csv"))
        if stop("svo"):
            print(f"[{method}] stop after svo  ({time.time()-t0:.1f}s)")
            return

    if "extract" in stages:
        if start_idx <= stages.index("extract"):
            if method in ("P_per-clr", "P_per-com"):
                df = await persona_claim_extract(texts, llm, out,
                                                 cfg["extract"]["personas"],
                                                 cfg["extract"]["text_truncate"])
            else:
                df = await claim_extract(texts, llm, out, cfg["extract"]["text_truncate"])
        else:
            df = pd.read_csv(os.path.join(out, "all_claims.csv"))
        if stop("extract"):
            print(f"[{method}] stop after extract  ({time.time()-t0:.1f}s)")
            return

    if "cluster" in stages:
        text_col = "triplet" if "triplet" in df.columns else "claim"
        items = df[text_col].astype(str).tolist()
        if start_idx <= stages.index("cluster"):
            labels, emb = embed_and_cluster(items, cfg, output_dir=out)
            np.save(os.path.join(out, "cluster_labels.npy"), labels)
            df["cluster_id"] = labels
            df.to_csv(os.path.join(out, "claims_clustered.csv"), index=False)
        else:
            labels = np.load(os.path.join(out, "cluster_labels.npy"))
            emb = np.load(os.path.join(out, "embeddings.npy"))
        if method == "P_SVO":
            groups = centroid_per_cluster(items, labels, emb)
        else:
            groups = group_by_label(items, labels)
        if stop("cluster"):
            print(f"[{method}] stop after cluster  ({time.time()-t0:.1f}s)")
            return

    if "decompose" in stages:
        if method == "P_com":
            items_for_decomp = texts
        else:
            items_for_decomp = [{"text_hash_id": r["claim_hash"], "text": r["claim"]}
                                for _, r in df.iterrows()]
        if start_idx <= stages.index("decompose"):
            decomp = await text_decompose(items_for_decomp, llm, out)
        else:
            with open(os.path.join(out, "text_decomposition.jsonl")) as f:
                decomp = [json.loads(line) for line in f if line.strip()]
        if stop("decompose"):
            print(f"[{method}] stop after decompose  ({time.time()-t0:.1f}s)")
            return

    if "graph" in stages:
        if start_idx <= stages.index("graph"):
            G = await build_graph(decomp, llm, out)
        else:
            with open(os.path.join(out, "graph.pkl"), "rb") as f:
                G = pickle.load(f)
        if stop("graph"):
            print(f"[{method}] stop after graph  ({time.time()-t0:.1f}s)")
            return

    if "attributes" in stages:
        if start_idx <= stages.index("attributes"):
            G = await gen_attributes(G, llm, out)
        else:
            with open(os.path.join(out, "graph.pkl"), "rb") as f:
                G = pickle.load(f)
        if stop("attributes"):
            print(f"[{method}] stop after attributes  ({time.time()-t0:.1f}s)")
            return

    if "leiden" in stages:
        if start_idx <= stages.index("leiden"):
            communities = leiden_communities(G, out, **cfg["leiden"])
        else:
            communities = _load_communities(out)
        if method == "P_com":
            groups = community_content_raw(communities, G, out,
                                           cfg["synthesize"]["max_content_chars"])
        else:
            groups = community_content_backjoin(communities, df, out,
                                                cfg["synthesize"]["max_claims_per_cluster"])
        if stop("leiden"):
            print(f"[{method}] stop after leiden  ({time.time()-t0:.1f}s)")
            return

    if "synthesize" in stages:
        if method == "P_SVO":
            from io_utils import _write_summaries, sha256_hash, IDCounter
            counter = IDCounter("gs")
            summaries = [
                {"human_readable_id": counter.next(),
                 "title": t, "description": t,
                 "hash_id": sha256_hash(t), "community_id": cid}
                for cid, t in sorted(groups.items())
            ]
            _write_summaries(summaries, METHOD_LABELS[method], cfg["run_label"], out)
        elif method == "P_dinam":
            summaries = await synthesize_dinam_narrative(
                groups, llm, METHOD_LABELS[method], out, cfg["run_label"])
        else:
            summaries = await synthesize_superclaim(
                groups, llm, METHOD_LABELS[method], out, cfg["run_label"])
        print(f"[{method}] done in {time.time()-t0:.1f}s → {len(summaries)} group summaries")
        return summaries


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is None:
            continue
        if "." in k:
            section, key = k.split(".", 1)
            cfg.setdefault(section, {})[key] = v
        else:
            cfg[k] = v
    if not cfg.get("run_label"):
        base = os.path.splitext(os.path.basename(cfg["input_texts"]))[0]
        cfg["run_label"] = f"{cfg['method']}_{base}"
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--method", choices=list(METHOD_LABELS))
    p.add_argument("--input-texts")
    p.add_argument("--run-label")
    p.add_argument("--output-dir")
    p.add_argument("--skip-until", help="Resume from this stage (load earlier outputs from disk)")
    p.add_argument("--stop-after", help="Exit after this stage completes (frees GPU between phases)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, {
        "method": args.method,
        "input_texts": args.input_texts,
        "run_label": args.run_label,
        "output_dir": args.output_dir,
        "skip_until": args.skip_until,
        "stop_after": args.stop_after,
    })
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
