"""
Preprocess Davis and KIBA datasets into encoded .pkl files.
Handles space-separated data without headers.
"""

import os
import pandas as pd
from src.data.preprocess import build_vocab, encode_sequence


def preprocess_csv(input_csv: str, output_pkl: str, max_smiles_len: int, max_prot_len: int):
    print(f"\n📂 Processing {input_csv} ...")

    # Read as space-separated file without header
    df = pd.read_csv(input_csv, sep=r"\s+", header=None)

    # Assign correct column names
    df.columns = ["drug_id", "protein_id", "smiles", "protein_sequence", "affinity"]
    print(f"✅ Loaded {len(df)} samples | Columns: {list(df.columns)}")

    # Drop ID columns (not needed for model)
    df = df[["smiles", "protein_sequence", "affinity"]]

    # Build vocabularies
    smiles_vocab = build_vocab(df["smiles"].tolist())
    prot_vocab = build_vocab(df["protein_sequence"].tolist())
    print(f"🔡 SMILES vocab size: {len(smiles_vocab)} | Protein vocab size: {len(prot_vocab)}")

    # Encode and pad
    df["smiles_encoded"] = df["smiles"].apply(lambda s: encode_sequence(s, smiles_vocab, max_smiles_len))
    df["protein_encoded"] = df["protein_sequence"].apply(lambda s: encode_sequence(s, prot_vocab, max_prot_len))

    # Save processed file
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    df.to_pickle(output_pkl)
    print(f"✅ Saved processed dataset → {output_pkl} | shape: {df.shape}")


if __name__ == "__main__":
    preprocess_csv(
        "data/raw/davis.csv",
        "data/processed/davis.pkl",
        max_smiles_len=85,
        max_prot_len=1200,
    )

    preprocess_csv(
        "data/raw/kiba.csv",
        "data/processed/kiba.pkl",
        max_smiles_len=100,
        max_prot_len=1000,
    )
