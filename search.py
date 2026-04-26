from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from train import train
from utils import load_checkpoint, save_json


def run_search(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    combinations = itertools.product(
        args.hidden_dims,
        args.learning_rates,
        args.weight_decays,
        args.activations,
    )

    for run_idx, (hidden_dim, lr, weight_decay, activation) in enumerate(combinations, start=1):
        run_dir = output_root / f"run_{run_idx:02d}_hd{hidden_dim}_{activation}_lr{lr}_wd{weight_decay}"
        namespace = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=str(run_dir),
            hidden_dim=hidden_dim,
            activation=activation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=lr,
            lr_decay=args.lr_decay,
            min_lr=args.min_lr,
            weight_decay=weight_decay,
            val_ratio=args.val_ratio,
            seed=args.seed + run_idx,
        )
        print(f"[Search] Running {run_dir.name}")
        train(namespace)
        checkpoint = load_checkpoint(run_dir / "best_model.npz")
        history = checkpoint["history"]
        best_val_acc = max(history["val_acc"])
        results.append(
            {
                "run_dir": str(run_dir),
                "hidden_dim": hidden_dim,
                "lr": lr,
                "weight_decay": weight_decay,
                "activation": activation,
                "best_val_acc": best_val_acc,
            }
        )

    results.sort(key=lambda item: item["best_val_acc"], reverse=True)
    save_json(output_root / "search_results.json", results)
    print("Top hyperparameter settings:")
    for row in results[:5]:
        print(row)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search for Fashion-MNIST MLP hyperparameters.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-root", type=str, default="outputs/search")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.05, 0.01])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-4])
    parser.add_argument("--activations", type=str, nargs="+", default=["relu"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run_search(build_argparser().parse_args())
