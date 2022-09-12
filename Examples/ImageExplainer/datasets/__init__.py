from .loader_encapsulation import LoaderEncapsulation
from .cats_and_dogs_dataset import CatDogDataset
from .fashion_mnist import FashionMNISTDataset
from .mnist import MnistDataset
from .mnist_and_fashionmnist import MNIST_and_FASHIONMNIST
from .mnist_background import MNISTImageBackground
from .mnist_noise import MNISTNoiseBackground
from .derma_panels import DermamnistPanel, Dermamnist
from .blood_panels import BloodMNIST, BloodMNISTPanel
from .autoencoder_dataset import AutoEncoderDataset
from .CelebA_dataset import CELEBA
from .CIFAR10_dataset import CIFAR10
from .simple_logistic_dataset import SimpleLogisticDataset


list_dataset = {
    "CatDogDataset": CatDogDataset,
    "FashionMNISTDataset": FashionMNISTDataset,
    "MnistDataset": MnistDataset,
    "MNISTImageBackground": MNISTImageBackground,
    "MNISTNoiseBackground": MNISTNoiseBackground,
    "DermamnistPanel": DermamnistPanel,
    "MNIST_and_FASHIONMNIST": MNIST_and_FASHIONMNIST,
    "Dermamnist": Dermamnist,
    "BloodMNIST": BloodMNIST,
    "BloodMNISTPanel": BloodMNISTPanel,
    "AutoEncoderDataset": AutoEncoderDataset,
    "CELEBA": CELEBA,
    "CIFAR10": CIFAR10,
    "SimpleLogisticDataset" : SimpleLogisticDataset,
}

class args_dataset_parameters():
    def __init__(self) -> None:
        self.root_dir = None
        self.batch_size_train = None
        self.batch_size_test = None
        self.noise_function = None
        self.download = None
        self.train_seed = None
        self.test_seed = None
        self.dataset = None
        self.target = None
        self.add_noise = False