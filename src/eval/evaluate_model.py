import torch
from tqdm import tqdm
import numpy as np
from src.models.deepdta_model import DeepDTA
from src.data.dataloader import get_dataloader
from src.eval.metrics import rmse, pearson_corr, concordance_index, plot_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_trained_model(model_path="models/deepdta_davis.pth", data_path="data/processed/davis.pkl"):
    print("🔍 Loading model and evaluating performance...")

    # Load data
    _, test_loader = get_dataloader(data_path, batch_size=64)

    # Load model
    model = DeepDTA(
        drug_vocab_size=65,
        prot_vocab_size=25,
        drug_max_len=85,
        prot_max_len=1200
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for smiles, protein, affinity in tqdm(test_loader, desc="Evaluating Model", ncols=100):
            smiles, protein = smiles.to(DEVICE), protein.to(DEVICE)
            preds = model(smiles, protein).squeeze().cpu().numpy()
            targets = affinity.numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    r = pearson_corr(all_targets, all_preds)
    rmse_val = rmse(all_targets, all_preds)
    ci = concordance_index(all_targets, all_preds)

    print(f"\n📊 Evaluation Results:")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"Pearson Correlation: {r:.4f}")
    print(f"Concordance Index: {ci:.4f}")

    # Save scatter plot
    plot_results(all_targets, all_preds)

if __name__ == "__main__":
    evaluate_trained_model()
