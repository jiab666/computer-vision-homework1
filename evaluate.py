from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autograd import cross_entropy_loss
from data_utils import CLASS_NAMES, batch_iterator, load_fashion_mnist
from mlp_model import MLPClassifier, MLPConfig
from utils import accuracy_score, confusion_matrix, load_checkpoint, save_json
from visualize import plot_confusion_matrix, plot_misclassified_samples


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    model_state = checkpoint["model_state"]
    model = MLPClassifier(
        MLPConfig(
            hidden_dim=int(model_state["hidden_dim"]),
            activation=str(model_state["activation"]),
            seed=int(model_state.get("seed", 42)),
        )
    )
    model.load_state_dict(model_state)

    dataset = load_fashion_mnist(root=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    losses = []
    predictions = []
    for batch_x, batch_y in batch_iterator(dataset["test_x"], dataset["test_y"], batch_size=args.batch_size, shuffle=False):
        logits = model.forward(batch_x)
        losses.append(float(cross_entropy_loss(logits, batch_y).data))
        predictions.append(logits.data.argmax(axis=1))

    preds = np.concatenate(predictions, axis=0)
    test_loss = float(np.mean(losses))
    test_acc = accuracy_score(dataset["test_y"], preds)
    cm = confusion_matrix(dataset["test_y"], preds, num_classes=len(CLASS_NAMES))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    plot_misclassified_samples(dataset["test_x"], dataset["test_y"], preds, output_dir / "misclassified_samples.png")
    save_json(
        output_dir / "metrics.json",
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "confusion_matrix": cm.tolist(),
        },
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Confusion matrix:")
    print(cm)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained NumPy MLP on Fashion-MNIST.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
