from nn_builder.toolkit.data_base import DataModule
import torchvision
from torchvision import transforms
import torch
from nn_builder.toolkit.utils import show_images

class FashionMNIST(DataModule):

    def __init__(self, batch_size=64, resize = (28,28)):
        super().__init__()
        self.save_hyperparameters()

        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])

        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)

        self.val = torchvision.datasets.FashionMNIST(
            root= self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)