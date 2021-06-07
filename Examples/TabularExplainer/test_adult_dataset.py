import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *




from torch.distributions import *
from torch.optim import *

import numpy as np

from functools import partial


if __name__ == "__main__":



    dataset_class = AdultDataset
    adult = AdultDatasetEncapsulation(dataset_class,128,1000)

    input_size_classifier = adult.get_shape()
    input_size_destructor = adult.get_shape()

    kernel_patch = (1,1)
    stride_patch = (1,1)


    classifier_no_var = ClassifierModel(input_size_classifier, adult.get_category(), middle_size=20)
    imputation_method = ConstantImputation(input_size=input_size_classifier)
    classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation_method)



    destructor_no_var = Destructor(input_size_destructor)
    destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization)
    trainer_no_var = noVariationalTraining(classification_no_var,
        destruction_no_var,
        kernel_patch = kernel_patch,
        stride_patch = stride_patch)

    
    temperature = 0.5
    optim_classification = Adam(classification_no_var.parameters())
    optim_destruction = Adam(destruction_no_var.parameters())

    for k in range(5):
        trainer_no_var.train(k, adult, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
                            lambda_reg= 0.1)

        trainer_no_var.test_no_var(adult, partial(RelaxedBernoulli,temperature))
        temperature *=0.5
    
    data, target= next(iter(adult.test_loader))
    data = data[:2]
    target = target[:2]
    sample_list = trainer_no_var.MCMC(adult,data,target,partial(RelaxedBernoulli,temperature),5000)

    show_interpretation_tabular(sample_list, data, target, adult)