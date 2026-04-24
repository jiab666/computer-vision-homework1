import math
from typing import Callable, Iterable, Optional, Set

import numpy as np


def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if requires_grad else None
        self._prev: Set[Tensor] = set(_children)
        self._backward: Callable[[], None] = lambda: None
        self._op = _op

    def __hash__(self) -> int:
        return id(self)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar tensors")
            grad = np.ones_like(self.data, dtype=np.float32)

        topo = []
        visited = set()

        def build(node: "Tensor") -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = grad.astype(np.float32)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad and other.grad is not None:
                other.grad += _sum_to_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += _sum_to_shape(other.data * out.grad, self.data.shape)
            if other.requires_grad and other.grad is not None:
                other.grad += _sum_to_shape(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad @ other.data.T
            if other.requires_grad and other.grad is not None:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def pow(self, exponent: float):
        out = Tensor(self.data**exponent, requires_grad=self.requires_grad, _children=(self,), _op="pow")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += (exponent * self.data ** (exponent - 1.0)) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in sorted((a if a >= 0 else a + self.data.ndim) for a in axes):
                        grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = math.prod(self.data.shape[a] for a in axes)
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(denom))

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        sigma = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sigma, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += sigma * (1.0 - sigma) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += (1.0 - value**2) * out.grad

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = 1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / exp_shifted.sum(axis=axis, keepdims=True)
        out = Tensor(np.log(probs + 1e-12), requires_grad=self.requires_grad, _children=(self,), _op="log_softmax")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                grad = out.grad
                summed = grad.sum(axis=axis, keepdims=True)
                self.grad += grad - probs * summed

        out._backward = _backward
        return out

    def __getitem__(self, index):
        out = Tensor(self.data[index], requires_grad=self.requires_grad, _children=(self,), _op="slice")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                np.add.at(self.grad, index, out.grad)

        out._backward = _backward
        return out


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    log_probs = logits.log_softmax(axis=1)
    batch_indices = np.arange(targets.shape[0])
    selected = log_probs[batch_indices, targets]
    return -(selected.mean())
