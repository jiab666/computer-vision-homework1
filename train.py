from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autograd import cross_entropy_loss
from data_utils import batch_iterator, load_fashion_mnist
from mlp_model import MLPClassifier, MLPConfig
from optim import SGD
from utils import accuracy_score, load_checkpoint, save_checkpoint, save_json
from visualize import plot_training_curves, visualize_first_layer_weights


def evaluate_split(model: MLPClassifier, x: np.ndarray, y: np.ndarray, batch_size: int = 512) -> tuple[float, float]:
    losses = []
    predictions = []
    for batch_x, batch_y in batch_iterator(x, y, batch_size=batch_size, shuffle=False):
        logits = model.forward(batch_x)
        loss = cross_entropy_loss(logits, batch_y)
        losses.append(float(loss.data))
        predictions.append(logits.data.argmax(axis=1))
    preds = np.concatenate(predictions, axis=0)
    return float(np.mean(losses)), accuracy_score(y, preds)


def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    dataset = load_fashion_mnist(root=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)

    model = MLPClassifier(
        MLPConfig(
            hidden_dim=args.hidden_dim,
            activation=args.activation,
            seed=args.seed,
        )
    )
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_path = output_dir / "best_model.npz"

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in batch_iterator(
            dataset["train_x"], dataset["train_y"], batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch
        ):
            model.zero_grad()
            logits = model.forward(batch_x)
            loss = cross_entropy_loss(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.data))
            batch_preds = logits.data.argmax(axis=1)
            train_correct += int((batch_preds == batch_y).sum())
            train_total += int(batch_y.shape[0])

        train_loss = float(np.mean(train_losses))
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate_split(model, dataset["val_x"], dataset["val_y"], batch_size=args.eval_batch_size)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.lr)

        print(
            f"Epoch {epoch:03d} | lr={optimizer.lr:.5f} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_path,
                {
                    "model_state": model.state_dict(),
                    "history": history,
                    "mean": dataset["mean"],
                    "std": dataset["std"],
                },
            )

        new_lr = optimizer.lr * args.lr_decay
        optimizer.set_lr(max(new_lr, args.min_lr))

    best_checkpoint = load_checkpoint(best_path)
    save_json(
        output_dir / "train_config.json",
        {
            "hidden_dim": args.hidden_dim,
            "activation": args.activation,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_decay": args.lr_decay,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "best_val_acc": best_val_acc,
        },
    )
    plot_training_curves(history, output_dir)
    visualize_first_layer_weights(best_checkpoint["model_state"]["fc1.weight"], output_dir / "first_layer_weights.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best checkpoint to: {best_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on Fashion-MNIST.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/default_run")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--activation", type=str, choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
