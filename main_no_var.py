from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

from functools import partial

if __name__ == "__main__":
    mnist = DatasetMnist(64,1000)

    # classifier_no_var = ClassifierModel(28*28, mnist.get_category())
    classifier_no_var = ConvClassifier(1)
    imputation_method = LearnImputation()
    classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation_method)
    # destructor_no_var = Destructor(28*28)
    destructor_no_var = ConvDestructor(1)
    destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization)
    trainer_no_var = noVariationalTraining(classification_no_var, destruction_no_var)
    temperature = 0.5
    optim_classification = Adam(classification_no_var.parameters())
    optim_destruction = Adam(destruction_no_var.parameters())

    trainer_no_var.train(0, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
                        lambda_reg= 0)
    trainer_no_var.test_no_var(mnist, partial(RelaxedBernoulli,temperature))
    
    data, target= next(iter(mnist.test_loader))
    data = data[:2]
    target = target[:2]
    sample_list = trainer_no_var.MCMC(mnist,data,target,partial(RelaxedBernoulli,temperature),5000)

    show_interpretation(sample_list, data, target)

    