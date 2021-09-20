import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
sys.path.append("/home/hhjs/MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *
from interpretation_regression import *


from torch.distributions import *
from torch.optim import *
import torch
from functools import partial

def get_default():
    

    args_output = {}
    args_output["path"] = "D:\\DTU\\firstProject\\MissingDataResults\\TestwithNoise" # Path to results
    args_output["experiment_name"] = "all_z"

    args_dataset = {}
    # args_dataset["dataset"] = LinearSeparableDataset
    args_dataset["dataset"] = CircleDataset
    args_dataset["loader"] = LoaderArtificial
    args_dataset["root_dir"] = None
    args_dataset["batch_size_train"] = 128
    args_dataset["batch_size_test"] = 64

    # Case for numerical point :
    args_dataset["nb_shape"] = 10
    args_dataset["nb_dim"] = 4
    args_dataset["ratio_sigma"] = 0.5
    args_dataset["sigma"] = 1.0
    args_dataset["prob_simplify"] = 0.5
    args_dataset["give_index"]= True
    args_dataset["nb_samples_train"] = 1000
    args_dataset["nb_samples_test"] = 200
    args_dataset["generate_new"] = False
    args_dataset["save"] = False
    args_dataset["centroids_path"] = None

    args_classification = {}

    args_classification["input_size_classification_module"] = (1,4) # Size before imputation
    args_classification["input_size_classifier"] = (1,4) # Size after imputation
    args_classification["input_size_classifier_baseline"] = (1,4) # Size before imputation (should be size of data)


    args_classification["classifier"] = ClassifierModel
    args_classification["classifier_baseline"] = None


    args_classification["imputation"] = ConstantImputation
    args_classification["cste_imputation"] = 0
    args_classification["add_mask"] = False


    args_classification["autoencoder"] = AutoEncoder # Autoencoder Network to use
    args_classification["post_process_regularization"] = DatasetBasedImputation # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_classification["reconstruction_regularization"] = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_classification["lambda_reconstruction"] = 0.01 # Parameter for controlling the reconstruction regularization
    args_classification["train_postprocess"] = False # If true, free the parameters of autoencoder during the training (loss guided by classification)
    args_classification["train_reconstruction_regularization"] = False # If true, free the parameters of autoencoder during the training (loss guided by a reconstruction loss)
    args_classification["noise_function"] = DropOutNoise(pi = 0.3) # Noise used to pretrain the autoencoder
    args_classification["nb_imputation"]= 1

    args_destruct = {}

    args_destruct["input_size_destructor"] = (1,4)
    args_destruct["input_size_autoencoder"] = (1,4)


  
    args_destruct["regularization"] = free_regularization
    args_destruct["lambda_regularisation"] = 1.0 # Entre 1 et 10 maintenant
    args_destruct["destructor"] = Destructor
    args_destruct["destructor"] = DestructorSimpleV2
    args_destruct["regularization_var"] = free_regularization
    args_destruct["lambda_regularization_var"] = 0.0
    args_destruct["destructor_var"] = None #DestructorSimilarVar
    args_destruct["kernel_patch"] = (1,1)
    args_destruct["stride_patch"] = (1,1)

    
    args_complete_trainer = {}
    args_complete_trainer["complete_trainer"] = noVariationalTraining # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args_complete_trainer["feature_extractor"] = None

    args_train = {}
    args_train["nb_epoch"] = 15 # Training the complete model
    args_train["nb_epoch_pretrain_autoencoder"] = 10 # Training the complete model
    args_train["nb_epoch_pretrain"] = 0 # Training auto encoder
    args_train["Nexpectation_train"] = 10 # Number K in the IWAE-similar loss 

    args_train["sampling_distribution_train"] = Bernoulli # If using reparametrization (ie noVariationalTraining), need rsample
    args_train["sampling_distribution_train_var"] = Bernoulli
    args_train["temperature_train_init"] = 1.0
    args_train["temperature_decay"] = 0.5
    args_train["use_cuda"] = torch.cuda.is_available()


    args_train["optim_classification"] = partial(Adam, lr=1e-4) #Learning rate for classification module
    args_train["optim_destruction"] = partial(Adam, lr=1e-4) # Learning rate for destruction module
    args_train["optim_destruction_var"] = partial(Adam, lr=1e-4) # Learning rate for the variationnal destruction module used in Variationnal Training
    args_train["optim_feature_extractor"] = partial(Adam, lr=1e-4) # Learning rate for the feature extractor if any
    args_train["optim_baseline"] = partial(Adam, lr=1e-4) # Learning rate for the baseline network
    args_train["optim_autoencoder"] = partial(Adam, lr=1e-4)

    
    args_test = {}
    args_test["sampling_distribution_test"] = Bernoulli # Sampling distribution used during test 
    args_test["temperature_test"] = 0.001
    args_test["Nexpectation_test"] = 10

    return  args_output, args_dataset, args_classification, args_destruct, args_complete_trainer, args_train, args_test
