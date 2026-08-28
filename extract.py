import asyncio
import os

import pandas as pd

from io_utils import sha256_hash, save_jsonl
from llm import ClaimList, TextDecomposition
from prompts import (
    CLAIM_EXTRACTION,
    PERSONA_CLAIM_EXTRACTION,
    PERSONAS,
    TEXT_DECOMPOSITION,
)


def svo_extract(texts, output_dir, spacy_model="en_core_web_lg",
                max_triplets_per_text=20, char_truncate=10_000):
    import spacy
    nlp = spacy.load(spacy_model, disable=["ner", "textcat", "lemmatizer"])
    nlp.max_length = 50_000

    rows = []
    truncated = [t["text"][:char_truncate] for t in texts]
    for doc, t in zip(nlp.pipe(truncated, batch_size=1000), texts):
        n = 0
        for sent in doc.sents:
            if n >= max_triplets_per_text:
                break
            for token in sent:
                if token.pos_ != "VERB":
                    continue
                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass", "agent")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "prep", "pobj", "acomp", "oprd")]
                if not subjects:
                    continue
                for subj in subjects:
                    s_text = " ".join(x.text for x in subj.subtree)
                    v = token.lemma_
                    if objects:
                        for obj in objects:
                            o_text = " ".join(x.text for x in obj.subtree)
                            rows.append({"text_hash_id": t["text_hash_id"], "triplet": f"{s_text} {v} {o_text}"})
                            n += 1
                    else:
                        comp = " ".join(x.text for x in token.subtree if x != token and x not in subj.subtree)
                        if comp:
                            rows.append({"text_hash_id": t["text_hash_id"], "triplet": f"{s_text} {v} {comp}"})
                            n += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "svo_triplets.csv"), index=False)
    print(f"Extracted {len(df)} SVO triplets from {len(texts)} texts.")
    return df


async def claim_extract(texts, llm, output_dir, char_truncate=50_000):
    rows = []

    async def process(item):
        query = CLAIM_EXTRACTION.format(text=item["text"][:char_truncate])
        resp = await llm.call(query, response_format=ClaimList)
        if not resp:
            return
        for c in resp.get("claims", [])[:10]:
            c = str(c).strip()
            if c:
                rows.append({"text_hash_id": item["text_hash_id"], "claim": c})

    await asyncio.gather(*[process(t) for t in texts])
    df = pd.DataFrame(rows).drop_duplicates(subset=["claim"]).reset_index(drop=True)
    df["claim_hash"] = df["claim"].map(sha256_hash)
    df.to_csv(os.path.join(output_dir, "all_claims.csv"), index=False)
    print(f"Extracted {len(df)} unique claims.")
    return df


async def persona_claim_extract(texts, llm, output_dir, personas, char_truncate=50_000):
    rows = []

    async def process(item, pkey):
        query = PERSONA_CLAIM_EXTRACTION.format(text=item["text"][:char_truncate])
        resp = await llm.call(query, response_format=ClaimList, system_prompt=PERSONAS[pkey])
        if not resp:
            return
        for c in resp.get("claims", [])[:3]:
            c = str(c).strip()
            if c:
                rows.append({"text_hash_id": item["text_hash_id"], "persona": pkey, "claim": c})

    await asyncio.gather(*[process(t, p) for t in texts for p in personas])
    df = pd.DataFrame(rows).drop_duplicates(subset=["claim"]).reset_index(drop=True)
    df["claim_hash"] = df["claim"].map(sha256_hash)
    df.to_csv(os.path.join(output_dir, "all_claims.csv"), index=False)
    print(f"Extracted {len(df)} unique claims across {len(personas)} personas.")
    return df


async def text_decompose(texts, llm, output_dir):
    results = []

    async def process(item):
        query = TEXT_DECOMPOSITION.format(text=item["text"])
        resp = await llm.call(query, response_format=TextDecomposition)
        if resp:
            results.append({"text_hash_id": item["text_hash_id"], "text": item["text"], "response": resp})

    await asyncio.gather(*[process(t) for t in texts])
    save_jsonl(results, os.path.join(output_dir, "text_decomposition.jsonl"))
    print(f"Decomposed {len(results)} texts.")
    return results
