from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

if __name__ == "__main__":
    mnist = DatasetMnist(64,1000)

    classifier = ClassifierModel(28, mnist.get_category())
    classification_vanilla = ClassificationModule(classifier)
    optim_classifier = Adam(classification_vanilla.parameters())
    trainer_vanilla = ordinaryTraining(classification_vanilla)
    trainer_vanilla.train(0, mnist,optim_classifier)
    trainer_vanilla.test(mnist)





