import argparse
import asyncio
import json
import os
from collections import Counter
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from llm import LLMClient


SYSTEM = (
    "You are an expert annotator labeling narrative claims. "
    "Pick exactly one of the four categories. "
    "Prefer Other over forcing a topical match when off-topic. "
    "Prefer Noise over Other when the text is not a coherent claim."
)

DEFS = {
    "ukraine": (
        "Claim primarily about the Russia's war in Ukraine or directly related geopolitics: "
        "Putin, Zelensky, Kremlin, Russian military, NATO in the war context, Ukrainian "
        "government/military, sanctions, war-driven refugee or energy crises."
    ),
    "climate": (
        "Claim primarily about climate change, climate science, climate policy, "
        "renewable energy, fossil fuels as climate issue, net-zero, environmental "
        "regulation, or climate-denial narratives."
    ),
    "other": (
        "Coherent claim on a clearly different topic: US domestic politics unrelated "
        "to the war, COVID, migration not tied to the war, gender, religion, Big Tech, "
        "celebrity, sport, finance unrelated to the war."
    ),
    "noise": (
        "Not a coherent claim: fragment, repeated token, placeholder, gibberish, "
        "meta-commentary, or title/description pair without propositional content."
    ),
}

LABELS = ("ukraine", "climate", "other", "noise")


def variant_a(text):
    return f"""\
Classify the following narrative claim into ONE category.

- A. War in Ukraine — {DEFS['ukraine']}
- B. Climate — {DEFS['climate']}
- C. Other — {DEFS['other']}
- D. Noise — {DEFS['noise']}

Decision rules:
1. Use "primarily about" as the test.
2. If a claim spans both climate and Ukraine, pick the dominant topic; if balanced, pick A.
3. If unsure between a topical label and C, pick C.
4. If the text is not a coherent claim, pick D regardless of topic.

Claim:
\"\"\"{text}\"\"\"
""", {"A": "ukraine", "B": "climate", "C": "other", "D": "noise"}


def variant_b(text):
    return f"""\
Classify the following narrative claim into ONE category.

- A. Climate — {DEFS['climate']}
- B. Other — {DEFS['other']}
- C. Noise — {DEFS['noise']}
- D. War in Ukraine — {DEFS['ukraine']}

Decision rules: same as default but with this letter mapping.

Claim:
\"\"\"{text}\"\"\"
""", {"A": "climate", "B": "other", "C": "noise", "D": "ukraine"}


def variant_c(text):
    return f"""\
Classify the following narrative claim into ONE category.

- "ukraine" — {DEFS['ukraine']}
- "climate" — {DEFS['climate']}
- "other" — {DEFS['other']}
- "noise" — {DEFS['noise']}

Claim:
\"\"\"{text}\"\"\"
""", {l: l for l in LABELS}


VARIANTS = {"A": variant_a, "B": variant_b, "C": variant_c}


class LetterLabel(BaseModel):
    label: Literal["A", "B", "C", "D"]
    confidence: Literal["high", "medium", "low"]
    rationale: str


class StringLabel(BaseModel):
    label: Literal["ukraine", "climate", "other", "noise"]
    confidence: Literal["high", "medium", "low"]
    rationale: str


async def classify_one(llm, text, variant_key, k_self_consistency, temperature):
    builder = VARIANTS[variant_key]
    prompt, mapping = builder(text)
    schema = StringLabel if variant_key == "C" else LetterLabel
    coros = [llm.call(prompt, response_format=schema, system_prompt=SYSTEM) for _ in range(k_self_consistency)]
    results = await asyncio.gather(*coros)
    labels = [mapping[r["label"]] for r in results if r and r.get("label") in mapping]
    if not labels:
        return None
    counter = Counter(labels)
    label, votes = counter.most_common(1)[0]
    return {"label": label, "agreement": votes / k_self_consistency, "samples": labels}


async def classify_all(llm, texts, variants, k_self_consistency, temperature):
    out = []
    for i, text in enumerate(texts):
        per_variant = {}
        for v in variants:
            per_variant[v] = await classify_one(llm, text, v, k_self_consistency, temperature)
        majority = Counter(p["label"] for p in per_variant.values() if p)
        final = majority.most_common(1)[0][0] if majority else "noise"
        out.append({"text": text, "label": final, "variants": per_variant})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)} classified")
    return out


async def main_async(args):
    df = pd.read_csv(os.path.join(args.run_dir, "group_summaries.csv"))
    texts = df["description"].astype(str).tolist()
    print(f"Classifying {len(texts)} group summaries…")

    llm = LLMClient(provider=args.provider, model=args.model, api_key=args.api_key,
                    api_base=args.api_base, concurrency=args.concurrency,
                    max_tokens=512, temperature=args.temperature)

    variants = list(args.variants)
    results = await classify_all(llm, texts, variants, args.k, args.temperature)

    df["topic"] = [r["label"] for r in results]
    out_path = os.path.join(args.run_dir, "topic_per_item.csv")
    df.to_csv(out_path, index=False)
    print(f"Per-item: {out_path}")

    counts = Counter(df["topic"])
    summary = {l: {"count": counts[l], "share": round(counts[l] / len(df), 4)} for l in LABELS}
    with open(os.path.join(args.run_dir, "topic_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--variants", nargs="+", default=["A", "B", "C"], choices=list(VARIANTS))
    p.add_argument("--k", type=int, default=5, help="Self-consistency K samples per variant")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--provider", default="vllm", choices=["vllm", "openai"])
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--api-base", default="http://127.0.0.1:11434/v1")
    p.add_argument("--api-key", default="your-secret-key")
    p.add_argument("--concurrency", type=int, default=100)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
