import asyncio
import hashlib
import json
import os

import pandas as pd

from llm import GroupSummaryResponse, NarrativeResponse
from prompts import DINAM_NARRATIVE, SUPERCLAIM


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_input_texts(path):
    df = pd.read_csv(path)
    items = []
    for _, row in df.iterrows():
        text = str(row["text"])
        thid = str(row["text_hash_id"]) if "text_hash_id" in df.columns and pd.notna(row.get("text_hash_id")) else sha256_hash(text)
        items.append({"text_hash_id": thid, "text": text})
    return items


class IDCounter:
    def __init__(self, prefix, start=0):
        self.prefix = prefix
        self.count = start

    def next(self):
        self.count += 1
        return f"{self.prefix}_{self.count}"


async def synthesize_superclaim(groups, llm, method, output_dir, run_label):
    print(f"Synthesizing {len(groups)} groups with super-claim prompt...")
    raw = []

    async def synth(cid, content):
        resp = await llm.call(SUPERCLAIM.format(content=content),
                              response_format=GroupSummaryResponse)
        if resp:
            raw.append({"community_id": int(cid), "response": resp})

    await asyncio.gather(*[synth(cid, c) for cid, c in groups.items()])
    save_jsonl(raw, os.path.join(output_dir, "community_summaries.jsonl"))

    counter = IDCounter("gs")
    summaries = []
    for r in sorted(raw, key=lambda x: x["community_id"]):
        for elem in r["response"].get("group_summaries", []):
            desc = elem.get("description", "")
            if not desc:
                continue
            summaries.append({
                "human_readable_id": counter.next(),
                "title": elem.get("title", ""),
                "description": desc,
                "hash_id": sha256_hash(desc),
                "community_id": r["community_id"],
            })

    return _write_summaries(summaries, method, run_label, output_dir)


async def synthesize_dinam_narrative(groups, llm, method, output_dir, run_label):
    print(f"Synthesizing {len(groups)} groups with DiNaM narrative prompt...")
    raw = []

    async def synth(cid, content):
        resp = await llm.call(DINAM_NARRATIVE.format(claims=content),
                              response_format=NarrativeResponse)
        if resp:
            raw.append({"community_id": int(cid), "narrative": resp.get("response", "")})

    await asyncio.gather(*[synth(cid, c) for cid, c in groups.items()])
    save_jsonl(raw, os.path.join(output_dir, "community_summaries.jsonl"))

    counter = IDCounter("gs")
    summaries = []
    for r in sorted(raw, key=lambda x: x["community_id"]):
        n = r["narrative"]
        if not n:
            continue
        summaries.append({
            "human_readable_id": counter.next(),
            "title": n,
            "description": n,
            "hash_id": sha256_hash(n),
            "community_id": r["community_id"],
        })

    return _write_summaries(summaries, method, run_label, output_dir)


def _write_summaries(summaries, method, run_label, output_dir):
    pd.DataFrame(summaries).to_csv(os.path.join(output_dir, "group_summaries.csv"), index=False)
    save_json({
        "label": run_label,
        "method": method,
        "n_summaries": len(summaries),
        "topic_descriptions": {s["human_readable_id"]: s["description"] for s in summaries},
    }, os.path.join(output_dir, "result.json"))
    print(f"Wrote {len(summaries)} group summaries.")
    return summaries
