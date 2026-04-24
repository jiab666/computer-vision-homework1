from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autograd import Tensor


@dataclass
class MLPConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 256
    output_dim: int = 10
    activation: str = "relu"
    seed: int = 42


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        limit = np.sqrt(2.0 / in_features)
        self.weight = Tensor(rng.normal(0.0, limit, size=(in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features), dtype=np.float32), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class MLPClassifier:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.fc1 = Linear(config.input_dim, config.hidden_dim, rng)
        self.fc2 = Linear(config.hidden_dim, config.hidden_dim, rng)
        self.fc3 = Linear(config.hidden_dim, config.output_dim, rng)

    def _activate(self, x: Tensor) -> Tensor:
        if self.config.activation == "relu":
            return x.relu()
        if self.config.activation == "sigmoid":
            return x.sigmoid()
        if self.config.activation == "tanh":
            return x.tanh()
        raise ValueError(f"Unsupported activation: {self.config.activation}")

    def forward(self, x: np.ndarray) -> Tensor:
        tensor_x = Tensor(x)
        hidden1 = self._activate(self.fc1(tensor_x))
        hidden2 = self._activate(self.fc2(hidden1))
        return self.fc3(hidden2)

    def parameters(self) -> list[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x).data
        return logits.argmax(axis=1)

    def state_dict(self) -> dict[str, np.ndarray | str | int]:
        return {
            "fc1.weight": self.fc1.weight.data.copy(),
            "fc1.bias": self.fc1.bias.data.copy(),
            "fc2.weight": self.fc2.weight.data.copy(),
            "fc2.bias": self.fc2.bias.data.copy(),
            "fc3.weight": self.fc3.weight.data.copy(),
            "fc3.bias": self.fc3.bias.data.copy(),
            "hidden_dim": self.config.hidden_dim,
            "activation": self.config.activation,
            "seed": self.config.seed,
        }

    def load_state_dict(self, state_dict: dict[str, np.ndarray | str | int]) -> None:
        self.fc1.weight.data = np.asarray(state_dict["fc1.weight"], dtype=np.float32)
        self.fc1.bias.data = np.asarray(state_dict["fc1.bias"], dtype=np.float32)
        self.fc2.weight.data = np.asarray(state_dict["fc2.weight"], dtype=np.float32)
        self.fc2.bias.data = np.asarray(state_dict["fc2.bias"], dtype=np.float32)
        self.fc3.weight.data = np.asarray(state_dict["fc3.weight"], dtype=np.float32)
        self.fc3.bias.data = np.asarray(state_dict["fc3.bias"], dtype=np.float32)
