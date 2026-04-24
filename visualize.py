from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_utils import CLASS_NAMES


def plot_training_curves(history: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_accuracy_curve.png", dpi=200)
    plt.close()


def visualize_first_layer_weights(weights: np.ndarray, output_path: str | Path, max_filters: int = 64) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = weights.T[:max_filters].reshape(-1, 28, 28)
    cols = 8
    rows = int(np.ceil(len(filters) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for idx, filt in enumerate(filters):
        ax = axes[idx // cols, idx % cols]
        ax.imshow(filt, cmap="coolwarm")
        ax.axis("off")
    fig.suptitle("First-Layer Weights")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_misclassified_samples(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
    max_samples: int = 16,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrong_idx = np.where(labels != predictions)[0][:max_samples]
    if wrong_idx.size == 0:
        return

    cols = 4
    rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for plot_idx, data_idx in enumerate(wrong_idx):
        ax = axes[plot_idx // cols, plot_idx % cols]
        ax.imshow(images[data_idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{CLASS_NAMES[labels[data_idx]]}\nP:{CLASS_NAMES[predictions[data_idx]]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
