import torch.nn as nn

class DatasetBasedClassifier(nn.Module):
    def __init__(self, dataset):
        super(DatasetBasedClassifier, self).__init__()
        assert hasattr(dataset, "true_predictor")
        self.dataset = dataset


    def __call__(self, x):
        return self.dataset.true_predictor(x)
