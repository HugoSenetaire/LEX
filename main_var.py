from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

from functools import partial

if __name__ == "__main__":
    mnist = DatasetMnist(64,1000)

    # classifier_var = ClassifierModel(28*28, mnist.get_category())
    classifier_var = ConvClassifier(1)
    classification_var = ClassificationModule(classifier_var, imputation=ConstantImputation())
    # destructor = Destructor(28*28)
    destructor = ConvDestructor(1)
    destructor_var = ConvDestructorVar(1)
    # destructor_var = DestructorVariational(28*28, mnist.get_category())
    destruction_var = DestructionModule(destructor, destructorVar= destructor_var, regularization=free_regularization, regularization_var= free_regularization)
    trainer_var = variationalTraining(classification_var, destruction_var)
    temperature = torch.tensor([0.5])
    if torch.cuda.is_available():
        temperature = temperature.cuda()
    optim_classification = Adam(classification_var.parameters())
    optim_destruction = Adam(destruction_var.parameters())
    for k in range(5):
        trainer_var.train(0, mnist, optim_classification, optim_destruction,Bernoulli , partial(RelaxedBernoulli,temperature))
        trainer_var.test_no_var(mnist, Bernoulli)
        trainer_var.test_var(mnist, Bernoulli, partial(RelaxedBernoulli, temperature))
    
    data, target= next(iter(mnist.test_loader))
    data = data[:2]
    target = target[:2]
    sample_list = trainer_var.MCMC(mnist,data, target, Bernoulli,5000)
    show_interpretation(sample_list, data, target)
    sample_list = trainer_var.MCMC_var(mnist,data, target, Bernoulli, partial(RelaxedBernoulli, temperature),5000)
    show_interpretation(sample_list, data, target)


    