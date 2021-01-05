from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

from functools import partial

if __name__ == "__main__":

    
    input_size_classifier = (1,28,28)
    input_size_destructor = (1,28,28)
    # stride_patch = (2,2)
    # kernel_patch = (2,2)
    stride_patch = (1,1)
    kernel_patch = (1,1)

    # mnist = DatasetFoo(2,10, shape=(3,3), len=10)
    # classifier_no_var = ClassifierModel(3, mnist.get_category())
    # imputation_method = LearnImputation()
    # classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation_method)
    # destructor_no_var = Destructor(3)
    # destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization)
    # trainer_no_var = noVariationalTraining(classification_no_var, destruction_no_var)
    # temperature = 0.5
    # optim_classification = Adam(classification_no_var.parameters())
    # optim_destruction = Adam(destruction_no_var.parameters())

    # trainer_no_var.train(0, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
    #                     lambda_reg= 0)
    # trainer_no_var.test_no_var(mnist, partial(RelaxedBernoulli,temperature))




    # mnist = DatasetMnist(64,1000)
    mnist = DatasetMnistVariation(64,1000)
    # mnist = DatasetMnistVariation2(64, 1000)
    classifier_no_var = ClassifierModel(input_size_classifier, mnist.get_category())
    # classifier_no_var = ConvClassifier(1)
    # imputation_method = LearnImputation(isRounded=True)
    imputation_method = ConstantImputation(isRounded=False)
    classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation_method)
    destructor_no_var = Destructor(input_size_destructor)
    # destructor_no_var = ConvDestructor(1)
    destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization)
    trainer_no_var = noVariationalTraining(classification_no_var, destruction_no_var, kernel_patch = kernel_patch, stride_patch = stride_patch)
    temperature = 0.5
    optim_classification = Adam(classification_no_var.parameters(), lr =1e-4 )
    optim_destruction = Adam(destruction_no_var.parameters(), lr = 1e-4)
    for k in range(10):
        temperature *=0.5
        trainer_no_var.train(k, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
                            lambda_reg= 0)
        trainer_no_var.test_no_var(mnist, partial(RelaxedBernoulli,temperature))
    
    data, target= next(iter(mnist.test_loader))
    data = data[:2]
    target = target[:2]
    sample_list = trainer_no_var.MCMC(mnist,data,target,partial(RelaxedBernoulli,temperature),5000)

    show_interpretation(sample_list, data, target)

    