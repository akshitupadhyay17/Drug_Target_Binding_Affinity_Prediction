"""
Data preprocessing for Drug-Target Binding Affinity project.
This script:
1. Builds vocabularies for SMILES and protein sequences.
2. Encodes sequences into integers.
3. Pads/truncates to fixed lengths.
4. Saves processed dataset for model training.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List

# ---------- Vocab utilities ----------
def build_vocab(sequences: List[str], min_freq: int = 1) -> Dict[str, int]:
    """
    Build character-level vocabulary from a list of sequences.
    0 is reserved for PAD token.
    """
    counter = Counter("".join(sequences))
    vocab = {ch: idx + 1 for idx, (ch, freq) in enumerate(counter.items()) if freq >= min_freq}
    return vocab

def encode_sequence(seq: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """
    Encode a sequence into integers with padding/truncation.
    """
    encoded = [vocab.get(ch, 0) for ch in seq[:max_len]]  # unknowns -> 0
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded

# ---------- Preprocessing pipeline ----------
def preprocess_dataset(
    input_csv: str,
    output_csv: str,
    max_smiles_len: int = 100,
    max_prot_len: int = 1000,
):
    """
    Preprocess raw dataset CSV into encoded & padded sequences.

    Args:
        input_csv: Path to raw CSV (columns: smiles, protein, affinity).
        output_csv: Path to save processed CSV.
        max_smiles_len: Maximum length for SMILES strings.
        max_prot_len: Maximum length for protein sequences.
    """
    df = pd.read_csv(input_csv)

    # Build vocabs
    smiles_vocab = build_vocab(df["smiles"].tolist())
    prot_vocab = build_vocab(df["protein"].tolist())

    print(f"SMILES vocab size: {len(smiles_vocab)}")
    print(f"Protein vocab size: {len(prot_vocab)}")

    # Encode + pad
    df["smiles_encoded"] = df["smiles"].apply(lambda x: encode_sequence(x, smiles_vocab, max_smiles_len))
    df["protein_encoded"] = df["protein"].apply(lambda x: encode_sequence(x, prot_vocab, max_prot_len))

    # Save processed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_pickle(output_csv)  # saving as pickle for list columns

    print(f"Processed dataset saved to {output_csv}")

    return smiles_vocab, prot_vocab


if __name__ == "__main__":
    # Example run
    smiles_vocab, prot_vocab = preprocess_dataset(
        input_csv="data/raw/sample.csv",
        output_csv="data/processed/sample_processed.pkl",
        max_smiles_len=100,
        max_prot_len=1000,
    )
