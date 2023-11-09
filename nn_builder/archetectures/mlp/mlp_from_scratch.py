from nn_builder.toolkit.classification_base import Classifier
import torch
from torch import nn
from torch.nn import functional as F

class MLPScratch(Classifier):

    def __init__(self, num_inputs, num_outputs,num_hiddens, lr , sigma = 0.01) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens)*sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))

        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs)*sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))


    def forward(self,X):
        X = X.reshape(-1, self.num_inputs)
        H = self.relu(torch.matmul(X,self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2

    def relu(self,X):
        a = torch.zeros_like(X)
        return torch.max(X,a)

    def loss(self, y_hat, y, averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1,))
        return F.cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')
