import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *



from torch.distributions import *
from torch.optim import *

from functools import partial


if __name__ == "__main__":

    
    input_size_classifier = (1,28,28)
    input_size_destructor = (1,28,28)
    stride_patch = (1,1)
    kernel_patch = (1,1)

  


    dataset_class = MnistVariationFashion

    mnist = LoaderEncapsulation(dataset_class, 64, 1000)
    classifier_no_var = ClassifierModel(input_size_classifier, mnist.get_category())
    imputation_method = ConstantImputation(isRounded=False)
    classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation_method)

    destructor_no_var = Destructor(input_size_destructor)
    destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization)
    trainer_no_var = noVariationalTraining(classification_no_var, destruction_no_var, kernel_patch = kernel_patch, stride_patch = stride_patch)
    
    
    temperature = 1.0
    optim_classification = Adam(classification_no_var.parameters())
    optim_destruction = Adam(destruction_no_var.parameters())


    for k in range(1):
        trainer_no_var.train(k, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
                            lambda_reg= 0)
        trainer_no_var.test_no_var(mnist, partial(RelaxedBernoulli,temperature))
        temperature *=0.5
    
    data, target= next(iter(mnist.test_loader))
    data = data[:2]
    target = target[:2]
