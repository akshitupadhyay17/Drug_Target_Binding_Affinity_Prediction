import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def pearson_corr(y_true, y_pred):
    """Pearson correlation coefficient"""
    return pearsonr(y_true, y_pred)[0]

def concordance_index(y_true, y_pred):
    """Concordance index (CI)"""
    n = 0
    h_sum = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1.0
                if (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]) or \
                   (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]):
                    h_sum += 1.0
                elif y_pred[i] == y_pred[j]:
                    h_sum += 0.5
    return h_sum / n

def plot_results(y_true, y_pred, save_path="results/scatter_davis.png"):
    """Plot true vs predicted affinities"""
    import os
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10, color='blue')
    plt.xlabel("True Affinity")
    plt.ylabel("Predicted Affinity")
    plt.title("DeepDTA Predictions vs True Values (Davis)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path)
    print(f"✅ Scatter plot saved at {save_path}")
