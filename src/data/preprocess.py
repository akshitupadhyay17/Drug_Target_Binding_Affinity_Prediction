"""
Utility functions for preprocessing sequences:
- Building vocabularies for SMILES and protein sequences
- Encoding and padding them into fixed-length integer tensors
"""

def build_vocab(sequences):
    """
    Build a vocabulary dictionary from a list of sequences.
    Each unique character gets a unique integer ID.
    """
    unique_chars = sorted(set(''.join(sequences)))
    vocab = {ch: idx + 1 for idx, ch in enumerate(unique_chars)}  # start indexing at 1
    vocab['<PAD>'] = 0  # reserved for padding
    return vocab


def encode_sequence(sequence, vocab, max_len):
    """
    Encode a sequence (SMILES or protein) using the vocabulary.
    Pads with zeros ('<PAD>') up to max_len, truncates if longer.
    """
    encoded = [vocab.get(ch, 0) for ch in sequence]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded
