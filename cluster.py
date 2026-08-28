import asyncio
import math
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from io_utils import IDCounter, sha256_hash
from llm import RelationshipReconstruction
from prompts import ATTRIBUTE_GENERATION, RELATIONSHIP_RECONSTRUCTION


def embed(texts, model_name, device="cuda", batch_size=256, normalize=True):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    arr = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=normalize)
    return np.asarray(arr)


def umap_reduce(emb, n_components=256, n_neighbors=15, min_dist=0.0,
                metric="cosine", random_state=42):
    try:
        from cuml.manifold import UMAP
    except ImportError:
        from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                   min_dist=min_dist, metric=metric, random_state=random_state)
    out = reducer.fit_transform(emb)
    return np.asarray(out.get() if hasattr(out, "get") else out)


def hdbscan_cluster(reduced, min_cluster_size=25, min_samples=20):
    try:
        from cuml.cluster import HDBSCAN
    except ImportError:
        from hdbscan import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        metric="euclidean", cluster_selection_method="eom")
    out = clusterer.fit_predict(reduced)
    return np.asarray(out.get() if hasattr(out, "get") else out)


def embed_and_cluster(texts, cfg, output_dir=None):
    emb = embed(texts, cfg["embed"]["model"], cfg["embed"]["device"],
                cfg["embed"]["batch_size"], cfg["embed"]["normalize"])
    reduced = umap_reduce(emb, **cfg["umap"])
    labels = hdbscan_cluster(reduced, **cfg["hdbscan"])
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clustered {len(texts)} items into {n_clusters} clusters ({int((labels == -1).sum())} noise).")
    if output_dir is not None:
        np.save(os.path.join(output_dir, "embeddings.npy"), emb)
    return labels, emb


def group_by_label(items, labels, max_per_cluster=100):
    groups = {}
    for item, lbl in zip(items, labels):
        if lbl < 0:
            continue
        groups.setdefault(int(lbl), []).append(item)
    return {cid: "\n".join(f"- {c}" for c in items[:max_per_cluster])
            for cid, items in groups.items()}


def centroid_per_cluster(items, labels, embeddings):
    groups = {}
    for i, (item, lbl) in enumerate(zip(items, labels)):
        if lbl < 0:
            continue
        groups.setdefault(int(lbl), []).append((i, item))
    picks = {}
    for cid, members in groups.items():
        idxs = [i for i, _ in members]
        c = embeddings[idxs].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        sims = embeddings[idxs] @ c
        picks[cid] = members[int(np.argmax(sims))][1]
    return picks


def _relationship_hash(a, b):
    return sha256_hash("".join(sorted([a, b])))


async def build_graph(decompositions, llm, output_dir):
    G = nx.Graph()
    ent_counter = IDCounter("entity")
    su_counter = IDCounter("semantic_unit")
    rel_counter = IDCounter("relationship")
    entities, sus, rels = {}, {}, {}

    async def process(data):
        thid = data.get("text_hash_id")
        for out in data.get("response", {}).get("Output", []):
            su_text = out.get("semantic_unit", "")
            su_hash = sha256_hash(su_text)
            if G.has_node(su_hash):
                G.nodes[su_hash]["weight"] += 1
            else:
                G.add_node(su_hash, type="semantic_unit", weight=1)
                sus[su_hash] = {"context": su_text, "human_readable_id": su_counter.next(), "text_hash_id": thid}

            ent_hashes = []
            for ent in out.get("entities", []):
                eh = sha256_hash(ent)
                ent_hashes.append(eh)
                if G.has_node(eh):
                    G.nodes[eh]["weight"] += 1
                else:
                    G.add_node(eh, type="entity", weight=1)
                    entities[eh] = {"context": ent, "human_readable_id": ent_counter.next(), "text_hash_id": thid}

            for rel_str in out.get("relationships", []):
                parts = [p.strip() for p in rel_str.split(",")]
                if len(parts) != 3:
                    q = RELATIONSHIP_RECONSTRUCTION.format(relationship=rel_str)
                    resp = await llm.call(q, response_format=RelationshipReconstruction)
                    if not resp:
                        continue
                    parts = [resp.get("source", ""), resp.get("relationship", ""), resp.get("target", "")]
                if len(parts) != 3 or not all(parts):
                    continue
                src, rel_type, tgt = parts
                rel_text = f"{src} {rel_type} {tgt}"
                sh, th = sha256_hash(src), sha256_hash(tgt)
                rh = _relationship_hash(sh, th)
                for h, txt in [(sh, src), (th, tgt)]:
                    if not G.has_node(h):
                        G.add_node(h, type="entity", weight=1)
                        entities[h] = {"context": txt, "human_readable_id": ent_counter.next(), "text_hash_id": thid}
                        ent_hashes.append(h)
                if not G.has_node(rh):
                    G.add_node(rh, type="relationship", weight=1)
                    rels[rh] = {"context": rel_text, "human_readable_id": rel_counter.next(), "text_hash_id": thid}
                    G.add_edge(sh, rh, weight=1)
                    G.add_edge(rh, th, weight=1)

            for eh in ent_hashes:
                if G.has_edge(su_hash, eh):
                    G[su_hash][eh]["weight"] += 1
                else:
                    G.add_edge(su_hash, eh, weight=1)

    await asyncio.gather(*[process(d) for d in decompositions])

    pd.DataFrame([{"hash_id": h, **d, "type": "entity"} for h, d in entities.items()]).to_csv(
        os.path.join(output_dir, "entities.csv"), index=False)
    pd.DataFrame([{"hash_id": h, **d, "type": "semantic_unit"} for h, d in sus.items()]).to_csv(
        os.path.join(output_dir, "semantic_units.csv"), index=False)
    pd.DataFrame([{"hash_id": h, **d, "type": "relationship"} for h, d in rels.items()]).to_csv(
        os.path.join(output_dir, "relationships.csv"), index=False)

    print(f"Graph: {G.number_of_nodes()} nodes ({len(entities)}E / {len(sus)}SU / {len(rels)}R), "
          f"{G.number_of_edges()} edges.")
    with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    return G


def _important_nodes(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    n = G.number_of_nodes()
    avg_degree = sum(dict(G.degree()).values()) / n
    k = max(round(np.log(n) * avg_degree ** 0.5), 1)
    important = set()
    try:
        for node in nx.core.k_core(G, k=k).nodes():
            if G.nodes[node].get("type") == "entity" and G.nodes[node].get("weight", 0) > 1:
                important.add(node)
    except nx.NetworkXError:
        pass
    bc = nx.betweenness_centrality(G, k=min(10, n))
    avg_bc = sum(bc.values()) / max(len(bc), 1)
    scale = max(round(math.log10(max(len(bc), 2))), 1)
    for node, val in bc.items():
        if val > avg_bc * scale and G.nodes[node].get("type") == "entity" and G.nodes[node].get("weight", 0) > 1:
            important.add(node)
    return list(important)


async def gen_attributes(G, llm, output_dir):
    node_texts = {}
    for fname in ["entities.csv", "semantic_units.csv", "relationships.csv"]:
        df = pd.read_csv(os.path.join(output_dir, fname))
        for _, row in df.iterrows():
            node_texts[row["hash_id"]] = str(row["context"])

    important = _important_nodes(G)
    print(f"Generating attributes for {len(important)} important entities.")

    counter = IDCounter("attribute")
    records = []

    async def gen(node):
        ent_text = node_texts.get(node, node)
        su, rel = "", ""
        for nb in G.neighbors(node):
            t = node_texts.get(nb, "")
            if not t:
                continue
            nt = G.nodes[nb].get("type")
            if nt == "semantic_unit":
                su += t + "\n"
            elif nt == "relationship":
                rel += t + "\n"
        q = ATTRIBUTE_GENERATION.format(entity=ent_text, semantic_units=su, relationships=rel)
        resp = await llm.call(q)
        if resp:
            h = sha256_hash(resp)
            G.add_node(h, type="attribute", weight=1)
            G.add_edge(node, h, weight=1)
            records.append({"hash_id": h, "human_readable_id": counter.next(),
                            "context": resp, "source_entity": ent_text})

    await asyncio.gather(*[gen(n) for n in important])
    if records:
        pd.DataFrame(records).to_csv(os.path.join(output_dir, "attributes.csv"), index=False)
    print(f"Generated {len(records)} attributes.")
    with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    return G


CONTENT_NODE_TYPES = frozenset(("semantic_unit", "attribute"))


def leiden_communities(G, output_dir, partition_type="modularity", resolution=1.0, seed=42):
    import igraph as ig
    import leidenalg as la

    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    if partition_type == "cpm":
        partition = la.find_partition(G_ig, la.CPMVertexPartition,
                                      resolution_parameter=resolution, seed=seed)
    elif partition_type == "mod_rb":
        partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition,
                                      resolution_parameter=resolution, seed=seed)
    else:
        partition = la.find_partition(G_ig, la.ModularityVertexPartition, seed=seed)

    communities = []
    assignments = []
    for i, idx_list in enumerate(partition):
        members = [G_ig.vs[idx]["name"] for idx in idx_list
                   if G.nodes.get(G_ig.vs[idx]["name"], {}).get("type") in CONTENT_NODE_TYPES]
        if members:
            communities.append((i, members))
            for m in members:
                assignments.append({"community_id": i, "node_hash_id": m,
                                    "node_type": G.nodes[m]["type"]})

    pd.DataFrame(assignments).to_csv(os.path.join(output_dir, "community_assignments.csv"), index=False)
    print(f"Leiden ({partition_type}): {len(communities)} communities.")
    return communities


def community_content_raw(communities, G, output_dir, max_chars=15_000):
    node_texts = {}
    for fname in ["semantic_units.csv", "attributes.csv"]:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            for _, row in pd.read_csv(p).iterrows():
                node_texts[row["hash_id"]] = str(row["context"])

    groups = {}
    for cid, members in communities:
        content = ""
        for n in members:
            if G.nodes.get(n, {}).get("type") not in CONTENT_NODE_TYPES:
                continue
            t = node_texts.get(n, "")
            if not t:
                continue
            content += t + "\n"
            if len(content) >= max_chars:
                content = content[:max_chars]
                break
        if content.strip():
            groups[cid] = content
    return groups


def community_content_backjoin(communities, claims_df, output_dir, max_claims=100):
    sus_df = pd.read_csv(os.path.join(output_dir, "semantic_units.csv"))
    claim_hash_to_text = dict(zip(claims_df["claim_hash"], claims_df["claim"]))
    su_hash_to_claim_hash = dict(zip(sus_df["hash_id"], sus_df["text_hash_id"]))

    groups = {}
    for cid, members in communities:
        seen = set()
        for n in members:
            ch = su_hash_to_claim_hash.get(n)
            if ch and ch in claim_hash_to_text:
                seen.add(ch)
        claim_strs = [claim_hash_to_text[h] for h in seen][:max_claims]
        if claim_strs:
            groups[cid] = "\n".join(f"- {c}" for c in claim_strs)
    return groups
