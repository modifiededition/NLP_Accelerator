import torch
from nn_builder.toolkit.models_base import Module
from nn_builder.optimizers.sgd import SGD

class LinearRegressionScratch(Module):
    def __init__(self, num_inputs, lr, sigma = 0.01) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self,X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y)**2 / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
