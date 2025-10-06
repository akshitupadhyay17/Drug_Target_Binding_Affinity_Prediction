import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDTA(nn.Module):
    def __init__(self,
                 drug_vocab_size: int,
                 prot_vocab_size: int,
                 drug_max_len: int,
                 prot_max_len: int,
                 emb_dim: int = 128,
                 num_filters: int = 32,
                 fc_dim: int = 1024,
                 dropout: float = 0.1):
        super(DeepDTA, self).__init__()

        # --- Drug branch ---
        self.drug_emb = nn.Embedding(drug_vocab_size + 1, emb_dim)
        self.drug_conv1 = nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=4)
        self.drug_conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=6)
        self.drug_conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 3, kernel_size=8)

        # --- Protein branch ---
        self.prot_emb = nn.Embedding(prot_vocab_size + 1, emb_dim)
        self.prot_conv1 = nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=4)
        self.prot_conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=6)
        self.prot_conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 3, kernel_size=8)

        # --- Fully connected layers ---
        combined_dim = (num_filters * 3) * 2  # Drug + Protein feature concatenation
        self.fc1 = nn.Linear(combined_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 512)
        self.out = nn.Linear(512, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, drug, protein):
        # Drug path
        d = self.drug_emb(drug).permute(0, 2, 1)  # [B, emb_dim, L]
        d = self.relu(self.drug_conv1(d))
        d = self.relu(self.drug_conv2(d))
        d = self.relu(self.drug_conv3(d))
        d = F.max_pool1d(d, kernel_size=d.shape[2]).squeeze(2)  # [B, num_filters*3]

        # Protein path
        p = self.prot_emb(protein).permute(0, 2, 1)
        p = self.relu(self.prot_conv1(p))
        p = self.relu(self.prot_conv2(p))
        p = self.relu(self.prot_conv3(p))
        p = F.max_pool1d(p, kernel_size=p.shape[2]).squeeze(2)

        # Combine
        combined = torch.cat((d, p), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = self.out(x)
        return output
    