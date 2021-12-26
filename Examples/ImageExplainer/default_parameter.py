import sys


sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
sys.path.append("/home/hhjs/MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *


from torch.distributions import *
from torch.optim import *
import torch
from functools import partial



def get_default():


    args_output = {}
    args_output["path"] = "C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\" # Path to results
    args_output["experiment_name"] = "all_z"




    args_complete_trainer = {}
    args_complete_trainer["complete_trainer"] = REINFORCE # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args_complete_trainer["save_every_epoch"] = 1
    args_complete_trainer["baseline"] = None
    args_complete_trainer["reshape_mask_function"] = simple_reshape
    args_complete_trainer["comply_with_dataset"] = True


    args_dataset = {}
    # args_dataset["dataset"] = LinearSeparableDataset
    args_dataset["dataset"] = MnistDataset
    args_dataset["loader"] = LoaderEncapsulation
    args_dataset["root_dir"] = os.path.join(args_output["path"], "datasets")
    args_dataset["batch_size_train"] = 128
    args_dataset["batch_size_test"] = 128
    args_dataset["noise_function"] = None
    args_dataset["download"] = True



    args_classification = {}
    args_classification["input_size_classification_module"] = (1,28,28) # Size before imputation
    args_classification["classifier"] = ClassifierLVL3

    args_classification["imputation"] = ConstantImputation
    args_classification["cste_imputation"] = 0
    args_classification["add_mask"] = False

    args_classification["post_process_network"] = None # Autoencoder Network to use
    args_classification["trainable_post_process"] = False # If true, free the parameters of network during the training (loss guided by classification)
    args_classification["pretrain_post_process"] = False # If true, pretrain the autoencoder with the training data
    args_classification["post_process_regularization"] = None # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_classification["reconstruction_regularization"] = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_classification["lambda_reconstruction"] = 0.01 # Parameter for controlling the reconstruction regularization
    args_classification["train_reconstruction_regularization"] = False # If true, free the parameters of autoencoder during the training (loss guided by a reconstruction loss)
    args_classification["noise_function"] = DropOutNoise(pi = 0.3) # Noise used to pretrain the autoencoder
    args_classification["nb_imputation"] = 1


    args_destruction = {}

    args_destruction["input_size_destructor"] = (1,28,28)
    args_destruction["output_size_destructor"] = (1,28,28)
    args_destruction["destructor"] = Destructor
    args_destruction["destructor_var"] = None #DestructorSimilarVar
    args_destruction["activation"] = torch.nn.LogSigmoid()
    # args_destruction["activation"] = torch.nn.LogSoftmax(dim=-1)

    # For regularization :
    args_destruction["trainable_regularisation"] = False
    args_destruction["regularization"] = LossRegularization
    args_destruction["lambda_reg"] = 0.0 # Entre 1 et 10 maintenant
    args_destruction["rate"] = 0.1
    args_destruction["loss_regularization"] = "L1" # L1, L2 
    args_destruction["batched"] = False


    args_destruction["regularization_var"] = LossRegularization
    args_destruction["lambda_regularization_var"] = 0.0
    args_destruction["rate_var"] = 0.1
    args_destruction["loss_regularization_var"] = "L1"
    args_destruction["batched_var"] = False



    args_distribution_module = {}
    args_distribution_module["distribution_module"] = DistributionModule
    args_distribution_module["distribution"] = Bernoulli
    args_distribution_module["distribution_relaxed"] = None
    args_distribution_module["temperature_init"] = 1.0
    args_distribution_module["test_temperature"] = 0.0
    args_distribution_module["scheduler_parameter"] = None
    args_distribution_module["sampling_subset_size"] = 2 # Sampling size for the subset 
    args_distribution_module["sampling_threshold"] = 0.5 # threshold for the selection






    args_train = {}
    # args_train["nb_epoch"] = 500 # Training the complete model
    args_train["nb_epoch"] = 15 # Training the complete model
    args_train["nb_epoch_post_hoc"] = 0 # Training the complete model
    args_train["nb_epoch_pretrain_autoencoder"] = 10 # Training auto encoder
    args_train["nb_epoch_pretrain"] = 0 # Training the complete model 
    args_train["nb_sample_z_train"] = 10 # Number K in the IWAE-similar loss 
    args_train["print_every"] = 1

    args_train["sampling_subset_size"] = 2 # Sampling size for the subset 
    args_train["temperature_train_init"] = 1.0
    args_train["temperature_decay"] = 0.9
    args_train["use_cuda"] = torch.cuda.is_available()
    args_train["fix_classifier_parameters"] = False
    args_train["post_hoc"] = False
    args_train["argmax_post_hoc"] = False
    args_train["post_hoc_guidance"] = None

    args_compiler = {}
    args_compiler["optim_classification"] = partial(Adam, lr=1e-3) #Learning rate for classification module
    args_compiler["optim_destruction"] = partial(Adam, lr=1e-3) # Learning rate for destruction module
    args_compiler["optim_destruction_var"] = partial(Adam, lr=1e-4) # Learning rate for the variationnal destruction module used in Variationnal Training
    args_compiler["optim_distribution_module"] = partial(Adam, lr=1e-4) # Learning rate for the feature extractor if any
    args_compiler["optim_baseline"] = partial(Adam, lr=1e-4) # Learning rate for the baseline network
    args_compiler["optim_autoencoder"] = partial(Adam, lr=1e-4)
    args_compiler["optim_post_hoc"] = partial(Adam, lr=1e-4)

    args_compiler["scheduler_classification"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6) #Learning rate for classification module
    args_compiler["scheduler_destruction"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6) # Learning rate for destruction module
    args_compiler["scheduler_destruction_var"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6) # Learning rate for the variationnal destruction module used in Variationnal Training
    args_compiler["scheduler_distribution_module"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6) # Learning rate for the feature extractor if any
    args_compiler["scheduler_baseline"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6) # Learning rate for the baseline network
    args_compiler["scheduler_autoencoder"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6)
    args_compiler["scheduler_post_hoc"] = partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma = 0.6)
    
    args_test = {}
    args_test["temperature_test"] = 0.001
    args_test["nb_sample_z_test"] = 10

    return  args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler
