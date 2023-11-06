import torch

from nn_builder.toolkit.models_base import Module


class LinearRegression(Module):

    def __init__(self, lr) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def loss(self, y_hat, y):
        fn = torch.nn.MSELoss()
        return fn(y_hat, y)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)