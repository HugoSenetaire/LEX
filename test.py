from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

from functools import partial

if __name__ == "__main__":


    input_size_classifier = (1,3,3)
    input_size_destructor = (1,3,3)
    stride_patch = (1,1)
    kernel_patch = (1,1)
    foo = DatasetFoo(16, 1000)
    imputation_method = ConstantImputation(cste = 0, input_size = input_size_destructor, isRounded=False)




    # classifier_no_var = ClassifierModel(input_size_classifier, foo.get_category())
    # classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation_method)


    # destructor_no_var = Destructor(input_size_destructor)
    # destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization)
    
    
    # trainer_no_var = noVariationalTraining(classification_no_var, destruction_no_var, kernel_patch = kernel_patch, stride_patch = stride_patch)
    # temperature = 0.5
    # optim_classification = Adam(classification_no_var.parameters(), lr =1e-4 )
    # optim_destruction = Adam(destruction_no_var.parameters(), lr = 1e-3)

    # for k in range(20):
    #     print(imputation_method.get_learnable_parameter())
    #     temperature *=0.5
    #     trainer_no_var.train(k, foo, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
    #                         lambda_reg= 0)
    #     trainer_no_var.test_no_var(foo, partial(RelaxedBernoulli,temperature))
    #     print(imputation_method.get_learnable_parameter())
    
    # data, target= next(iter(foo.test_loader))
    # data = data[:2]
    # target = target[:2]
    # sample_list = trainer_no_var.MCMC(foo,data,target,partial(RelaxedBernoulli,temperature),10000)

    # show_interpretation(sample_list, data, target, shape=(1,3,3))




    classifier_var = ClassifierModel(input_size_classifier, foo.get_category())
    classification_var = ClassificationModule(classifier_var, imputation=imputation_method)
    destructor_no_var = Destructor(input_size_destructor)
    destructor_var = DestructorVariational(input_size_destructor, foo.get_category())
    destruction_var = DestructionModule(destructor_no_var, destructor_var = destructor_var, regularization=free_regularization, regularization_var= free_regularization)
    trainer_var = variationalTraining(classification_var, destruction_var, stride_patch = stride_patch, kernel_patch=kernel_patch)
    temperature = torch.tensor([1.0])
    if torch.cuda.is_available():
        temperature = temperature.cuda()
    optim_classification = Adam(classification_var.parameters(), lr = 1e-4)
    optim_destruction = Adam(destruction_var.parameters(), lr = 1e-4)
    for k in range(10):
        temperature *= 0.5
        trainer_var.train(k, foo, optim_classification, optim_destruction,Bernoulli , partial(RelaxedBernoulli,temperature), lambda_reg=0.0, lambda_reg_var = 10.0)
        # trainer_var.test_no_var(foo, Bernoulli)
        # trainer_var.test_var(foo, Bernoulli, partial(RelaxedBernoulli, temperature))
        break
    
    # data, target= next(iter(foo.test_loader))
    # data = data[:2]
    # target = target[:2]
    # sample_list = trainer_var.MCMC(foo,data, target, Bernoulli,5000)
    # show_interpretation(sample_list, data, target, shape=(1,3,3))
    # sample_list = trainer_var.MCMC_var(foo,data, target, Bernoulli, partial(RelaxedBernoulli, temperature),5000)
    # show_interpretation(sample_list, data, target, shape=(1,3,3))
