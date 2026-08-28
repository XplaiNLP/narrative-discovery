import argparse
import asyncio
import csv
import json
import os
from collections import Counter
from typing import Literal

from pydantic import BaseModel, create_model

from llm import LLMClient
from prompts import PERSONAS  # the four pipeline personas — same as the persona pipelines

# Paper's two judge backbones (appendix H.2). Override with --models / --api-bases.
DEFAULT_MODELS = ["google/gemma-4-31B-it", "Qwen3.5-27B"]
DEFAULT_API_BASES = ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11435/v1"]


PROMPT = """\
You are annotating a narrative-to-taxonomy correspondence task.

An automated system extracted a narrative cluster from a corpus. Your job is to \
choose the single best taxonomy match for it from the candidates below — or to \
mark it as "Other" (in-domain but not represented in any candidate) or "None" \
(out-of-domain / incoherent).

Candidate order is randomized — do not infer ranking from position.

## Group summary
{description}

## Candidates
{candidates_block}

## Instructions
1. is_disinfo_narrative: yes / no — is this recognizable as a disinformation narrative?
2. label_match: pick exactly one of {choices_str}, "Other", "None".
3. confidence: 1 (lowest) to 5 (highest).
4. notes: one short sentence justifying your choice.
"""


def make_schema(top_k):
    cand_ids = tuple(f"cand_{i+1}" for i in range(top_k)) + ("Other", "None")
    return create_model(
        "Judgment",
        is_disinfo_narrative=(Literal["yes", "no"], ...),
        label_match=(Literal[cand_ids], ...),  # type: ignore
        confidence=(Literal[1, 2, 3, 4, 5], ...),
        notes=(str, ...),
    )


def detect_top_k(fieldnames):
    k = 0
    while f"cand_{k+1}_id" in fieldnames:
        k += 1
    return k


def build_prompt(row, top_k):
    lines = [
        f"  cand_{i}: [{row[f'cand_{i}_taxonomy']}/{row[f'cand_{i}_id']}] {row[f'cand_{i}_text']}"
        for i in range(1, top_k + 1)
    ]
    return PROMPT.format(
        description=row["gs_description"],
        candidates_block="\n".join(lines),
        choices_str=", ".join(f'"cand_{i+1}"' for i in range(top_k)),
    )


def model_label(model):
    """Short backbone label for output columns, e.g. google/gemma-4-31B-it -> gemma."""
    base = model.split("/")[-1].lower()
    for tag in ("gemma", "qwen", "llama", "mistral", "phi"):
        if tag in base:
            return tag
    return base.replace(".", "_").replace("-", "_")


async def annotate(rows, llm, persona_key, top_k):
    schema = make_schema(top_k)
    sys_prompt = PERSONAS[persona_key]
    out = [None] * len(rows)

    async def one(i, row):
        out[i] = await llm.call(build_prompt(row, top_k), response_format=schema,
                                system_prompt=sys_prompt)

    await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    return out


def majority_vote(judgments_by_judge):
    """Majority vote across an arbitrary set of judges (persona×model)."""
    n_rows = len(next(iter(judgments_by_judge.values())))
    out = []
    for i in range(n_rows):
        votes = {field: Counter() for field in ("is_disinfo_narrative", "label_match", "confidence")}
        notes = {}
        for jkey, judgments in judgments_by_judge.items():
            j = judgments[i] or {}
            for field in votes:
                if j.get(field) is not None:
                    votes[field][j[field]] += 1
            notes[jkey] = (j.get("notes") or "")
        merged = {f: (v.most_common(1)[0][0] if v else "") for f, v in votes.items()}
        merged["notes"] = " | ".join(f"{p}: {n}" for p, n in notes.items() if n)
        out.append(merged)
    return out


async def main_async(args):
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    top_k = detect_top_k(fieldnames)

    api_bases = args.api_bases
    if len(api_bases) == 1 and len(args.models) > 1:
        api_bases = api_bases * len(args.models)
    if len(api_bases) != len(args.models):
        raise SystemExit("--api-bases must have one entry, or one per --models entry")

    backbones = [(model_label(m), m, b) for m, b in zip(args.models, api_bases)]
    print(f"{len(rows)} rows, top-K={top_k}")
    print(f"panel: {len(PERSONAS)} personas x {len(backbones)} models = {len(PERSONAS)*len(backbones)} judges")
    print(f"  personas: {list(PERSONAS)}")
    print(f"  backbones: {[b[1] for b in backbones]}")

    # (label, persona) -> judgments; also track which judges belong to each backbone
    judgments = {}
    per_backbone_keys = {}
    for label, model, api_base in backbones:
        llm = LLMClient(provider=args.provider, model=model, api_key=args.api_key,
                        api_base=api_base, concurrency=args.concurrency,
                        max_tokens=512, temperature=0.7)
        per_backbone_keys[label] = []
        for pkey in PERSONAS:
            jkey = f"{label}_{pkey}"
            print(f"  judge: {jkey}")
            judgments[jkey] = await annotate(rows, llm, pkey, top_k)
            per_backbone_keys[label].append(jkey)

    # overall 8-judge panel majority + per-model 4-persona ensemble majorities
    panel = majority_vote(judgments)
    ensembles = {label: majority_vote({k: judgments[k] for k in keys})
                 for label, keys in per_backbone_keys.items()}

    out_fields = list(fieldnames)
    for col in ("is_disinfo_narrative", "label_match", "confidence", "notes"):
        if col not in out_fields:
            out_fields.append(col)
    for label in per_backbone_keys:  # per-model ensemble majority
        for sub in ("is_disinfo_narrative", "label_match", "confidence"):
            out_fields.append(f"{label}_maj_{sub}")
    for jkey in judgments:            # raw per-judge (persona x model)
        for sub in ("label_match", "confidence", "is_disinfo_narrative"):
            out_fields.append(f"{jkey}_{sub}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            o = dict(row)
            o.update(panel[i])
            for label, ens in ensembles.items():
                for sub in ("is_disinfo_narrative", "label_match", "confidence"):
                    o[f"{label}_maj_{sub}"] = ens[i].get(sub, "")
            for jkey, js in judgments.items():
                j = js[i] or {}
                for sub in ("label_match", "confidence", "is_disinfo_narrative"):
                    o[f"{jkey}_{sub}"] = j.get(sub, "")
            w.writerow(o)
    print(f"Wrote {args.output}")


def main():
    p = argparse.ArgumentParser(description="8-judge LLM panel (4 personas x 2 models), appendix H.2")
    p.add_argument("--input", required=True, help="CSV from human_val.py")
    p.add_argument("--output", required=True)
    p.add_argument("--provider", default="vllm", choices=["vllm", "openai"])
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="backbone models (paper: Gemma-4-31B-it + Qwen3.5-27B)")
    p.add_argument("--api-bases", nargs="+", default=DEFAULT_API_BASES,
                   help="one endpoint per model, or a single shared endpoint")
    p.add_argument("--api-key", default="your-secret-key")
    p.add_argument("--concurrency", type=int, default=100)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
