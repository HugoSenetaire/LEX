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
    # classifier_var = ClassifierModel(3, mnist.get_category())
    # classification_var = ClassificationModule(classifier_var, imputation=ConstantImputation())
    # destructor = Destructor(3)
    # # destructor = ConvDestructor(1)
    # # destructor_var = ConvDestructorVar(1)
    # destructor_var = DestructorVariational(3, mnist.get_category())
    # destruction_var = DestructionModule(destructor, destructorVar= destructor_var, regularization=free_regularization, regularization_var= free_regularization)
    # trainer_var = variationalTraining(classification_var, destruction_var)
    # temperature = torch.tensor([0.01])
    # if torch.cuda.is_available():
    #     temperature = temperature.cuda()
    # optim_classification = Adam(classification_var.parameters())
    # optim_destruction = Adam(destruction_var.parameters())
    # for k in range(1):
    #     trainer_var.train(0, mnist, optim_classification, optim_destruction,Bernoulli , partial(RelaxedBernoulli,temperature), Nexpectation=4)







    # mnist = DatasetMnist(64,1000)
    mnist = DatasetMnistVariation(64,1000)
    classifier_var = ClassifierModel(input_size_classifier, mnist.get_category())
    # classifier_var = ConvClassifier(1)
    classification_var = ClassificationModule(classifier_var, imputation=ConstantImputation(cste = 1000, isRounded=False))
    destructor = Destructor(input_size_destructor)
    # destructor = ConvDestructor(1)
    # destructor_var = ConvDestructorVar(1)
    # destructor_var = DestructorVariational(input_size_destructor, mnist.get_category())
    destructor_var = DestructorVariational(input_size_destructor, mnist.get_category())
    destruction_var = DestructionModule(destructor, destructor_var = destructor_var, regularization=free_regularization, regularization_var= free_regularization)
    trainer_var = variationalTraining(classification_var, destruction_var, stride_patch = stride_patch, kernel_patch=kernel_patch)
    temperature = torch.tensor([1.0])
    if torch.cuda.is_available():
        temperature = temperature.cuda()
    optim_classification = Adam(classification_var.parameters(), lr = 1e-4)
    optim_destruction = Adam(destruction_var.parameters(), lr = 1e-4)
    for k in range(10):
        temperature *= 0.5
        trainer_var.train(k, mnist, optim_classification, optim_destruction,Bernoulli , partial(RelaxedBernoulli,temperature), lambda_reg=0.0, lambda_reg_var = 1000.0)
        # trainer_var.train(k, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature) , partial(RelaxedBernoulli,temperature), lambda_reg=0.0)
        trainer_var.test_no_var(mnist, Bernoulli)
        trainer_var.test_var(mnist, Bernoulli, partial(RelaxedBernoulli, temperature))
    
    data, target= next(iter(mnist.test_loader))
    data = data[:2]
    target = target[:2]
    sample_list = trainer_var.MCMC(mnist,data, target, Bernoulli,5000)
    show_interpretation(sample_list, data, target)
    sample_list = trainer_var.MCMC_var(mnist,data, target, Bernoulli, partial(RelaxedBernoulli, temperature),5000)
    show_interpretation(sample_list, data, target)


    