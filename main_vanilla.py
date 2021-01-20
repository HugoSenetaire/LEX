from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

if __name__ == "__main__":

    input_size_classifier = (1, 28, 28)
    dataset_class = MnistVariation1
    mnist = LoaderEncapsulation(dataset_class,64,1000)

    classifier = ClassifierModel(input_size_classifier, mnist.get_category())
    classification_vanilla = ClassificationModule(classifier)
    optim_classifier = Adam(classification_vanilla.parameters())
    trainer_vanilla = ordinaryTraining(classification_vanilla)
    
    for epoch in range(10):
        trainer_vanilla.train(epoch, mnist,optim_classifier)
        trainer_vanilla.test(mnist)





