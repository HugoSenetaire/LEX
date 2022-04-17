import torch

##### ENCAPSULATION :

class LoaderArtificial():
    def __init__(self, dataset, batch_size_train = 512, batch_size_test = 512,):
        self.dataset = dataset
        self.dataset_train = self.dataset.dataset_train
        self.dataset_test = self.dataset.dataset_test
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                            batch_size=batch_size_train,
                            shuffle=True
                            )

        self.test_loader = torch.utils.data.DataLoader(
                            self.dataset_test,
                            batch_size=batch_size_test,
                            shuffle=False
                            )