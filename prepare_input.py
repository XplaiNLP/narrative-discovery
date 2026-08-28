import argparse
import hashlib
import os

import pandas as pd


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser(
        description="Convert any text-bearing CSV into the canonical input_texts.csv "
                    "(columns: text_hash_id, text) consumed by main.py.",
    )
    p.add_argument("src", help="Source CSV.")
    p.add_argument("dst", help="Destination CSV (will be created).")
    p.add_argument("--text-col", default="text",
                   help="Column in src that holds the document text. "
                        "For CO's all.csv use --text-col ad_creative_body.")
    p.add_argument("--min-length", type=int, default=20,
                   help="Drop rows whose stripped text is shorter than this. Default 20.")
    args = p.parse_args()

    df = pd.read_csv(args.src)
    if args.text_col not in df.columns:
        raise SystemExit(
            f"Column {args.text_col!r} not found in {args.src}. "
            f"Available: {df.columns.tolist()}"
        )

    df = df.rename(columns={args.text_col: "text"})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > args.min_length]
    df = df.reset_index(drop=True)

    df["text_hash_id"] = df["text"].map(sha256_hash)
    df = df.drop_duplicates(subset=["text_hash_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    df[["text_hash_id", "text"]].to_csv(args.dst, index=False)
    print(f"{args.src} -> {args.dst}: {len(df)} rows after filter/dedup")


if __name__ == "__main__":
    main()
