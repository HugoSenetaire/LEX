from .SupervizedLearningWithMissingData import *
from .classification_network import *
from .dataset_based_classifier import *
from .autoencoder_wrapper import *


classifiers_list = {
    "none": None,
    "ClassifierLinear" : ClassifierLinear,
    "ConvClassifier" : ConvClassifier,
    "ConvClassifier2" : ConvClassifier2,
    "ProteinCNN" : ProteinCNN,
    "ClassifierLvl1" : ClassifierLVL1,
    "ClassifierLvl2" : ClassifierLVL2,
    "ClassifierLvl3" : ClassifierLVL3,
    "RealXClassifier" : RealXClassifier,
    "DatasetBasedClassifier" : DatasetBasedClassifier,
    "AutoEncoderWrapper": AutoEncoderWrapper,
}


def get_pred_network(classifier_name):
    if classifier_name == "none" or classifier_name == None:
        return None
    elif classifier_name in classifiers_list:
        return classifiers_list[classifier_name]
    else:
        raise ValueError(f"Classifier {classifier_name} not found")

