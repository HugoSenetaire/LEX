from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

import numpy as np

from functools import partial


if __name__ == "__main__":

    # input_size_classifier = (20)
    input_size_classifier = (1, 28, 28)
    input_size_destructor = (1,28,28)
    # stride_patch = (2,2)
    # kernel_patch = (2,2)
    stride_patch = (1,1)
    kernel_patch = (1,1)



    dataset_class = MnistVariation1
    mnist = LoaderEncapsulation(dataset_class,64,1000)
    # mnist = DatasetMnistVariation(64,1000)
    # # mnist = DatasetMnistVariation2(64, 1000)
    # classifier_no_var = ClassifierModel(input_size_classifier, mnist.get_category(), middle_size=20)
    # imputation_network = imputationInvariantNetwork(input_size=(50), output_size=20)
    # imputation_method = PermutationInvariance(imputation_network, size_D=48, add_index=True)
    # # imputation_method = ConstantImputationRateReg()
    # # imputation_method = ConstantImputationInsideReg()
    # # imputation_method = ConstantImputationInsideReverseReg()


    noise_function = noise_gaussian
    mnist_noise = LoaderEncapsulation(dataset_class, 64, 1000, noisy = True, noise_function = noise_function)
    autoencoder_network = AutoEncoder().cuda()
    optim_autoencoder = Adam(autoencoder_network.parameters())


    data, target = next(iter(mnist_noise.test_loader))
    data = data[:2]
    target = target[:2]


    # show_interpretation(data.detach().cpu().numpy(), target.detach().cpu().numpy(), [0,0,0,0])

    # for k in range(5):
    #     train_autoencoder(autoencoder_network, mnist_noise, optim_autoencoder)
    #     test_autoencoder(autoencoder_network, mnist_noise)
    #     autoencoder_network.eval()
    #     output = autoencoder_network(data.cuda()).reshape(data.shape)


    # show_interpretation(output.detach().cpu().numpy(), target.detach().cpu().numpy(), [0,0,0,0])



    classifier_no_var = ClassifierModel(input_size_classifier, mnist.get_category(), middle_size=20)
    imputation_method = AutoEncoderImputation(autoencoder_network, to_train=True)
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
        trainer_no_var.train(k, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature),
                            lambda_reg= 0.1)

        # print(imputation_method.autoencoder.fc1.parameters())
        trainer_no_var.test_no_var(mnist, partial(RelaxedBernoulli,temperature))
        temperature *=0.5
    
    data, target= next(iter(mnist.test_loader))
    data = data[:2]
    target = target[:2]
    sample_list = trainer_no_var.MCMC(mnist,data,target,partial(RelaxedBernoulli,temperature),5000)

    show_interpretation(sample_list, data, target)