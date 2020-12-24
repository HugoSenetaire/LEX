import torch
import torchvision




class DatasetMnist():
    def __init__(self, batch_size_train, batch_size_test):
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader( torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
                            batch_size=batch_size_train, shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST('/files/', train=False, download=True,
                                                            transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(
                                                                (0.1307,), (0.3081,))
                                                            ])),
                            batch_size=batch_size_test, shuffle=True
                            )

    def get_category(self):
        return 10

    def get_shape(self):
        return (1,28,28)