"""Modul pro vizualizaci průběhu trénování a výsledků regrese."""

import os
from typing import List, Dict, Any, Optional
import numpy as np

# Zajištění headless režimu pro matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(history: List[Dict[str, Any]], output_path: str) -> None:
    """Vykreslí graf vývoje MSE a MAE v průběhu trénovacích epoch."""
    epochs = [h["epoch"] for h in history]
    train_mse = [h["train_mse"] for h in history]
    train_mae = [h["train_mae"] for h in history]
    test_mse = [h["test_mse"] for h in history if h.get("test_mse") is not None]
    test_mae = [h["test_mae"] for h in history if h.get("test_mae") is not None]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graf 1: MSE Loss
    ax1.plot(epochs, train_mse, label="Train MSE", color="#1f77b4", linewidth=2)
    if test_mse:
        ax1.plot(epochs[:len(test_mse)], test_mse, label="Test MSE", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_title("Vývoj střední kvadratické chyby (MSE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epocha", fontsize=11)
    ax1.set_ylabel("MSE [(mil. Kč)²]", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Graf 2: MAE
    ax2.plot(epochs, train_mae, label="Train MAE", color="#2ca02c", linewidth=2)
    if test_mae:
        ax2.plot(epochs[:len(test_mae)], test_mae, label="Test MAE", color="#d62728", linestyle="--", linewidth=2)
    ax2.set_title("Vývoj průměrné absolutní chyby (MAE)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epocha", fontsize=11)
    ax2.set_ylabel("MAE [mil. Kč]", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Graf učící křivky byl uložen do: {output_path}")


def plot_regression_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    a: float,
    b: float,
    test_metrics: Dict[str, float],
    output_path: str,
) -> None:
    """Vykreslí scatter plot dat spolu s naučenou regresní přímkou f(x) = ax + b."""
    plt.figure(figsize=(10, 6))
    
    # Body trénovací a testovací sady
    plt.scatter(X_train, y_train, color="#1f77b4", alpha=0.55, label=f"Trénovací data ({len(X_train)} vzorků)", s=35)
    plt.scatter(X_test, y_test, color="#ff7f0e", alpha=0.85, edgecolors="k", label=f"Testovací data ({len(X_test)} vzorků)", s=50)
    
    # Vykreslení regresní přímky
    x_min = min(float(np.min(X_train)), float(np.min(X_test))) * 0.95
    x_max = max(float(np.max(X_train)), float(np.max(X_test))) * 1.05
    x_line = np.linspace(x_min, x_max, 200)
    y_line = a * x_line + b
    
    label_line = (
        f"Regresní model: f(x) = {a:.4f} * x + {b:.4f}\n"
        f"Test MSE: {test_metrics['mse']:.4f} | Test MAE: {test_metrics['mae']:.4f} | R²: {test_metrics['r2']:.4f}"
    )
    plt.plot(x_line, y_line, color="#d62728", linewidth=2.5, label=label_line)
    
    plt.title("Jednoduchý lineární model: Predikce ceny bytu na základě plochy", fontsize=13, fontweight="bold")
    plt.xlabel("Plocha [m²]", fontsize=11)
    plt.ylabel("Cena [mil. Kč]", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Graf regresní přímky byl uložen do: {output_path}")
