
import torch


##### ENCAPSULATION :

class LoaderEncapsulation():
    def __init__(self, dataset, batch_size_train = 16, batch_size_test=500, transform =None, noise_function = None,):
        self.dataset = dataset
        self.dataset_test = self.dataset.dataset_test
        self.dataset_train = self.dataset.dataset_train
        self.dataset_val = self.dataset.dataset_val
        
        self.batch_size_test = batch_size_test
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_test
        
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=batch_size_train, shuffle=True, num_workers = 4)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=batch_size_test, shuffle=False, num_workers = 4)
        self.val_loader = torch.utils.data.DataLoader(self.dataset_val, batch_size=batch_size_test, shuffle=False, num_workers = 4)