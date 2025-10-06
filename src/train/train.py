import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.deepdta_model import DeepDTA
from src.data.dataloader import get_dataloader
import os

# ----------------------------
# Configuration
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-4
DATA_PATH = "data/processed/davis.pkl"   # change to kiba.pkl when needed
MODEL_SAVE_PATH = "models/deepdta_davis.pth"

os.makedirs("models", exist_ok=True)


# ----------------------------
# Training Function
# ----------------------------
def train_model(pkl_path=DATA_PATH):
    # Get train/test dataloaders
    train_loader, test_loader = get_dataloader(pkl_path, batch_size=BATCH_SIZE)

    # Initialize model
    model = DeepDTA(
        drug_vocab_size=65,     # depends on preprocessing vocab
        prot_vocab_size=25,
        drug_max_len=85,
        prot_max_len=1200,
        emb_dim=128,
        num_filters=32,
        fc_dim=1024,
        dropout=0.1
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 Training DeepDTA on {DEVICE} for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for smiles, protein, affinity in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100):
            smiles, protein, affinity = smiles.to(DEVICE), protein.to(DEVICE), affinity.to(DEVICE)

            optimizer.zero_grad()
            preds = model(smiles, protein).squeeze()
            loss = criterion(preds, affinity)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {avg_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training complete! Model saved to {MODEL_SAVE_PATH}")

    # Optional evaluation right after training
    evaluate_model(model, test_loader)


# ----------------------------
# Simple Evaluation Function
# ----------------------------
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for smiles, protein, affinity in tqdm(test_loader, desc="Evaluating", ncols=100):
            smiles, protein, affinity = smiles.to(DEVICE), protein.to(DEVICE), affinity.to(DEVICE)
            preds = model(smiles, protein).squeeze()
            loss = criterion(preds, affinity)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"\n🧪 Test Loss (MSE): {avg_test_loss:.4f}")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    train_model()