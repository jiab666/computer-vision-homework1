import gzip
import struct
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

import numpy as np


FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def ensure_fashion_mnist(root: str = "data") -> Path:
    data_dir = Path(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FASHION_MNIST_URLS.items():
        path = data_dir / Path(url).name
        if not path.exists():
            print(f"Downloading {path.name} ...")
            urlretrieve(url, path)
    return data_dir


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols).astype(np.float32)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num).astype(np.int64)


def load_fashion_mnist(root: str = "data", val_ratio: float = 0.1, seed: int = 42) -> dict[str, np.ndarray]:
    data_dir = ensure_fashion_mnist(root)
    train_x = _read_idx_images(data_dir / "train-images-idx3-ubyte.gz") / 255.0
    train_y = _read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_x = _read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz") / 255.0
    test_y = _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    rng = np.random.default_rng(seed)
    indices = rng.permutation(train_x.shape[0])
    val_size = int(train_x.shape[0] * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return {
        "train_x": train_x[train_idx],
        "train_y": train_y[train_idx],
        "val_x": train_x[val_idx],
        "val_y": train_y[val_idx],
        "test_x": test_x,
        "test_y": test_y,
        "mean": mean,
        "std": std,
    }


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(x.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]
