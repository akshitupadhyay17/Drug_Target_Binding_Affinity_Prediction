import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

class DrugTargetDataset(Dataset):
    """Custom PyTorch Dataset for Drug–Target Binding Affinity data."""
    def __init__(self, pkl_file):
        # Load preprocessed DataFrame
        self.data = pd.read_pickle(pkl_file)
        # Shuffle the data for good measure
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = torch.tensor(self.data.iloc[idx]["smiles_encoded"], dtype=torch.long)
        protein = torch.tensor(self.data.iloc[idx]["protein_encoded"], dtype=torch.long)
        affinity = torch.tensor(self.data.iloc[idx]["affinity"], dtype=torch.float32)
        return smiles, protein, affinity


def get_dataloader(pkl_path, batch_size=32, split_ratio=0.83, num_workers=0):
    """
    Returns PyTorch DataLoaders for train and test sets.
    - split_ratio: 0.83 means 5:1 split (train:test)
    """
    dataset = DrugTargetDataset(pkl_path)
    total_len = len(dataset)
    train_len = int(total_len * split_ratio)
    test_len = total_len - train_len

    train_ds, test_ds = random_split(dataset, [train_len, test_len],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataset: {pkl_path}")
    print(f"Train samples: {train_len} | Test samples: {test_len}")
    return train_loader, test_loader