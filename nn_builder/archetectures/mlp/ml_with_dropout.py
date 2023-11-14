import torch
from torch import nn
from nn_builder.toolkit.classification_base import Classifier
from torch.nn import functional as F

def dropout_layer(X, dropout):

    assert 0 <= dropout <= 1

    if dropout == 1:
        return torch.zeros_like(X)

    mask = (torch.rand(X.shape) > dropout).float()

    return (X * mask)/ (1- dropout)


class MLPDropoutScratch(Classifier):

    def __init__(self, num_hidden_1, num_hidden_2, num_outputs, dropout_1, dropout_2, lr) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.l1 = nn.LazyLinear(num_hidden_1)
        self.l2 = nn.LazyLinear(num_hidden_2)
        self.l3 = nn.LazyLinear(num_outputs)
        self.relu  = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.l1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)

        H2 = self.relu(self.l2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.l3(H2)

    def loss(self, y_hat, y,averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1,))
        return F.cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')


class MLPDropoutConcise(Classifier):
    def __init__(self, num_outputs, num_hidden_1, num_hidden_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hidden_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hidden_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))

    def loss(self, y_hat, y,averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1,))
        return F.cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')
