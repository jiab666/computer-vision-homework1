import numpy as np

from autograd import Tensor


class SGD:
    def __init__(self, parameters: list[Tensor], lr: float, weight_decay: float = 0.0) -> None:
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue
            grad = param.grad
            if self.weight_decay > 0.0 and param.data.ndim > 1:
                grad = grad + self.weight_decay * param.data
            param.data -= self.lr * grad

    def set_lr(self, lr: float) -> None:
        self.lr = lr
