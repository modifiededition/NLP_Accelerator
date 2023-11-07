from typing import Iterator
import torch
from nn_builder.toolkit.classification_base import Classifier
from torch.nn.parameter import Parameter


class LogisticRegressionScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma = 0.01) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self) -> Iterator[Parameter]:
        return [self.W, self.b]

    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return self.softmax(torch.matmul(X, self.W) + self.b)

    def loss(self, y_hat, y):
        return self.cross_entropy(y_hat, y)

    def cross_entropy(self, y_hat, y):
        l = -torch.log(y_hat[range(len(y_hat)),y]).mean()
        return l

    def softmax(self, X):
        X_exp = torch.exp(X)
        X_exp = X_exp / X_exp.sum(1, keepdims=True)
        return X_exp


