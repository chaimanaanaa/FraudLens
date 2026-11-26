"""
Prepare data from local raw datasets.

Reads:
  data/raw/smsspamcollection/SMSSpamCollection  — TSV: label<TAB>text
  data/raw/archive/Phishing_Email.csv           — CSV: varies by Kaggle version

Outputs:
  corpus/           — 50 .txt files for RAG (30 spam/phishing + 20 ham)
  data/test_messages/textes.jsonl      — 15 message texts
  data/test_messages/annotations.jsonl — 15 ground-truth labels

Usage:
  python scripts/prepare_data.py
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "corpus"
TEST_DIR = ROOT / "data" / "test_messages"
SMS_PATH = ROOT / "data" / "raw" / "smsspamcollection" / "SMSSpamCollection"
EMAIL_PATH = ROOT / "data" / "raw" / "archive" / "Phishing_Email.csv"

CORPUS_SPAM = 30
CORPUS_HAM = 20
TEST_SPAM = 10
TEST_HAM = 5

RANDOM_SEED = 42


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sms() -> pd.DataFrame:
    df = pd.read_csv(
        SMS_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df["label"] = df["label"].str.strip().str.lower()
    df["source"] = "sms"
    df["fraud_type"] = df["label"].map({"spam": "smishing", "ham": None})
    return df[["label", "text", "source", "fraud_type"]].dropna(subset=["text"])


def _detect_email_columns(df: pd.DataFrame):
    """
    Handle multiple Kaggle CSV column-name conventions:
      Version A: 'Email Text', 'Email Type'   (Safe Email / Phishing Email)
      Version B: 'text_combined', 'label'      (0=ham, 1=spam)
      Version C: 'body', 'label'
    Returns (text_col, label_col) and a label mapping dict.
    """
    cols = [c.lower().strip() for c in df.columns]
    col_map = {c.lower().strip(): c for c in df.columns}

    if "email text" in cols and "email type" in cols:
        return (
            col_map["email text"],
            col_map["email type"],
            {"phishing email": "spam", "safe email": "ham"},
        )
    if "text_combined" in cols and "label" in cols:
        return col_map["text_combined"], col_map["label"], {1: "spam", 0: "ham", "1": "spam", "0": "ham"}
    if "body" in cols and "label" in cols:
        return col_map["body"], col_map["label"], {1: "spam", 0: "ham", "1": "spam", "0": "ham"}
    if "text" in cols and "label" in cols:
        return col_map["text"], col_map["label"], {1: "spam", 0: "ham", "1": "spam", "0": "ham"}

    # Fallback: first two columns
    print(f"WARNING: Unknown CSV columns {list(df.columns)}. Using first two.")
    return df.columns[1], df.columns[0], {}


def load_emails() -> pd.DataFrame:
    df = pd.read_csv(EMAIL_PATH, encoding="utf-8", on_bad_lines="skip")
    text_col, label_col, label_map = _detect_email_columns(df)

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df["raw_label"] = df["raw_label"].apply(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

    if label_map:
        df["label"] = df["raw_label"].map(
            {str(k).lower(): v for k, v in label_map.items()}
        )
    else:
        df["label"] = df["raw_label"].apply(
            lambda x: "spam" if str(x) in {"1", "phishing email", "spam"} else "ham"
        )

    df["source"] = "email"
    df["fraud_type"] = df["label"].map({"spam": "phishing", "ham": None})
    return df[["label", "text", "source", "fraud_type"]].dropna(subset=["text", "label"])


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_stratified(df: pd.DataFrame, n_spam: int, n_ham: int, rng: random.Random) -> pd.DataFrame:
    spam = df[df["label"] == "spam"].copy()
    ham = df[df["label"] == "ham"].copy()

    spam = spam.sample(n=min(n_spam, len(spam)), random_state=RANDOM_SEED)
    ham = ham.sample(n=min(n_ham, len(ham)), random_state=RANDOM_SEED)

    return pd.concat([spam, ham]).reset_index(drop=True)


# ── Writers ───────────────────────────────────────────────────────────────────

def write_corpus(df: pd.DataFrame, corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing corpus files
    for f in corpus_dir.glob("*.txt"):
        f.unlink()

    for i, row in df.iterrows():
        filename = f"doc_{i:03d}_{row['label']}_{row['source']}.txt"
        content = f"# label={row['label']} source={row['source']}\n{clean_text(row['text'])}"
        (corpus_dir / filename).write_text(content, encoding="utf-8")

    print(f"Wrote {len(df)} corpus files to {corpus_dir}")


def write_test_set(df: pd.DataFrame, test_dir: Path) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)

    texts_path = test_dir / "textes.jsonl"
    annots_path = test_dir / "annotations.jsonl"

    with open(texts_path, "w", encoding="utf-8") as ft, \
         open(annots_path, "w", encoding="utf-8") as fa:
        for i, row in df.iterrows():
            msg_id = f"msg_{i:03d}"
            text_item = {
                "id": msg_id,
                "text": clean_text(row["text"]),
                "source": row["source"],
            }
            annot_item = {
                "id": msg_id,
                "label": row["label"],
                "fraud_type": row["fraud_type"] if pd.notna(row.get("fraud_type")) else None,
            }
            ft.write(json.dumps(text_item, ensure_ascii=False) + "\n")
            fa.write(json.dumps(annot_item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} test messages to {test_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)

    print("Loading SMS Spam Collection...")
    sms_df = load_sms()
    print(f"  SMS: {len(sms_df[sms_df['label']=='spam'])} spam, {len(sms_df[sms_df['label']=='ham'])} ham")

    print("Loading Phishing Email dataset...")
    email_df = load_emails()
    print(f"  Email: {len(email_df[email_df['label']=='spam'])} spam, {len(email_df[email_df['label']=='ham'])} ham")

    all_df = pd.concat([sms_df, email_df], ignore_index=True)

    # Deduplicate on text
    all_df = all_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # ── Corpus: 50 examples (30 spam + 20 ham) ───────────────────────────────
    corpus_df = sample_stratified(all_df, CORPUS_SPAM, CORPUS_HAM, rng)
    write_corpus(corpus_df, CORPUS_DIR)

    # ── Test set: 15 examples (10 spam + 5 ham), no overlap with corpus ───────
    corpus_texts = set(corpus_df["text"].tolist())
    remaining = all_df[~all_df["text"].isin(corpus_texts)].reset_index(drop=True)
    test_df = sample_stratified(remaining, TEST_SPAM, TEST_HAM, rng)
    write_test_set(test_df, TEST_DIR)

    print("\nSummary:")
    print(f"  Corpus: {len(corpus_df[corpus_df['label']=='spam'])} spam + {len(corpus_df[corpus_df['label']=='ham'])} ham = {len(corpus_df)} total")
    print(f"  Test:   {len(test_df[test_df['label']=='spam'])} spam + {len(test_df[test_df['label']=='ham'])} ham = {len(test_df)} total")
    print("Done.")


if __name__ == "__main__":
    main()
