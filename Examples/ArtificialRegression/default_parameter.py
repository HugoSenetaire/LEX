import sys
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")

from missingDataTrainingModule.classification_training import ordinaryTraining, EVAL_X
from missingDataTrainingModule.selection_training import selectionTraining
from missingDataTrainingModule.interpretation_training import SELECTION_BASED_CLASSIFICATION, REALX
from missingDataTrainingModule import PytorchDistributionUtils, utils_reshape, Classification, Selection
from datasets import *
import os


from torch.distributions import *
from torch.optim import *
import torch
from functools import partial

def get_dataset(args_dataset):
    dataset = args_dataset["dataset"](**args_dataset)

    loader = args_dataset["loader"](dataset, batch_size_train=args_dataset["batch_size_train"], batch_size_test=args_dataset["batch_size_test"],)
    return dataset, loader



def get_default():


    args_output = {}
    # args_output["path"] = "C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\" # Path to results
    args_output["path"] = "/scratch/hhjs" # Path to results

    args_output["experiment_name"] = "REINFORCE"




    args_complete_trainer = {}
    args_complete_trainer["complete_trainer"] = SELECTION_BASED_CLASSIFICATION
    args_complete_trainer["monte_carlo_gradient_estimator"] = PytorchDistributionUtils.gradientestimator.REINFORCE # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args_complete_trainer["save_every_epoch"] = 20
    args_complete_trainer["save_epoch_function"] = lambda epoch, nb_epoch: (epoch % args_complete_trainer["save_every_epoch"] == 0) or (epoch == nb_epoch-1) or (epoch<10)
    args_complete_trainer["baseline"] = None
    args_complete_trainer["reshape_mask_function"] = utils_reshape.collapse_in_batch
    args_complete_trainer["comply_with_dataset"] = True


    args_dataset = {}
    # args_dataset["dataset"] = LinearSeparableDataset
    args_dataset["dataset"] = DiagDataset
    args_dataset["loader"] = LoaderArtificial
    args_dataset["root_dir"] = os.path.join(args_output["path"], "datasets")
    args_dataset["batch_size_train"] = 1000
    args_dataset["batch_size_test"] = 1000
    args_dataset["noise_function"] = None
    args_dataset["cov"] = torch.tensor(1.0, dtype=torch.float32)
    args_dataset["covariance_type"] = "spherical" #Choice spherical diagonal full
    args_dataset["mean"] = torch.tensor(0.0, dtype=torch.float32)
    args_dataset["download"] = True
    args_dataset["dim_input"] = 11
    args_dataset["used_dim"] = 10



    args_classification = {}
    args_classification["input_size_classification_module"] = (1,2) # Size before imputation
    args_classification["classifier"] = Classification.classification_network.RealXClassifier

    args_classification["imputation"] = Classification.imputation.ConstantImputation
    args_classification["cste_imputation"] = 0
    args_classification["sigma_noise_imputation"] = 1.0
    args_classification["add_mask"] = False
    args_classification["module_imputation"] = None
    args_classification["nb_imputation_iwae"] = 1
    args_classification["nb_imputation_iwae_test"] = None #If none is given, turn to 1
    args_classification["nb_imputation_mc"] = 1
    args_classification["nb_imputation_mc_test"] = None #If none is given, turn to 1

    args_classification["reconstruction_regularization"] = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_classification["network_reconstruction"] = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_classification["lambda_reconstruction"] = 0.01 # Parameter for controlling the reconstruction regularization
    
    args_classification["post_process_regularization"] = None # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_classification["network_post_process"] = None # Autoencoder Network to use
    args_classification["post_process_trainable"] = False # If true, pretrain the autoencoder with the training data
    
    args_classification["mask_reg"] = None
    args_classification["mask_reg_rate"] = 0.5

    args_selection = {}

    args_selection["input_size_selector"] = (1,2)
    args_selection["output_size_selector"] = (1,2)
    args_selection["selector"] = Selection.selective_network.RealXSelector
    args_selection["selector_var"] = None 
    args_selection["activation"] = torch.nn.LogSigmoid()
    # args_selection["activation"] = torch.nn.LogSoftmax(dim=-1)

    # For regularization :
    args_selection["trainable_regularisation"] = False
    args_selection["regularization"] = Selection.regularization_module.LossRegularization
    args_selection["lambda_reg"] = 0.0 
    args_selection["rate"] = 0.0
    args_selection["loss_regularization"] = "L1" # L1, L2 
    args_selection["batched"] = False


    args_selection["regularization_var"] =  Selection.regularization_module.LossRegularization
    args_selection["lambda_regularization_var"] = 0.0
    args_selection["rate_var"] = 0.1
    args_selection["loss_regularization_var"] = "L1"
    args_selection["batched_var"] = False



    args_distribution_module = {}
    args_distribution_module["distribution_module"] = PytorchDistributionUtils.wrappers.DistributionModule
    args_distribution_module["distribution"] = Bernoulli
    args_distribution_module["distribution_relaxed"] = RelaxedBernoulli
    args_distribution_module["temperature_init"] = 1.0
    args_distribution_module["test_temperature"] = 1e-5
    args_distribution_module["scheduler_parameter"] = PytorchDistributionUtils.wrappers.regular_scheduler
    args_distribution_module["sampling_subset_size"] = 2 # Sampling size for the subset 
    args_distribution_module["sampling_threshold"] = 0.5 # threshold for the selection
    args_distribution_module["antitheis_sampling"] = False 


    
    args_classification_distribution_module = {}
    args_classification_distribution_module["distribution_module"] = PytorchDistributionUtils.wrappers.FixedBernoulli
    args_classification_distribution_module["distribution"] = Bernoulli
    args_classification_distribution_module["distribution_relaxed"] = RelaxedBernoulli
    args_classification_distribution_module["temperature_init"] = 0.1
    args_classification_distribution_module["test_temperature"] = 1e-5
    args_classification_distribution_module["scheduler_parameter"] = PytorchDistributionUtils.wrappers.regular_scheduler
    args_classification_distribution_module["sampling_subset_size"] = 2 # Sampling size for the subset 
    args_classification_distribution_module["sampling_threshold"] = 0.5 # threshold for the selection
    args_classification_distribution_module["antitheis_sampling"] = False 



    args_train = {}
    # args_train["nb_epoch"] = 500 # Training the complete model
    args_train["nb_epoch"] = 10 # Training the complete model
    args_train["nb_epoch_post_hoc"] = 0 # Training post_hoc
    args_train["nb_epoch_pretrain_selector"] = 0 # Pretrain selector
    args_train["use_regularization_pretrain_selector"] = False # Use regularization when pretraining the selector
    args_train["nb_epoch_pretrain"] = 0 # Training the complete model 
    args_train["nb_sample_z_train_monte_carlo"] = 1
    args_train["nb_sample_z_train_IWAE"] = 1  # Number K in the IWAE-similar loss
    args_train["loss_function"] = "NLL" # NLL, MSE

    args_train["training_type"] = "classic" # Options are ["classic", "alternate_ordinary", "alternate_fixing"]
    args_train["nb_step_fixed_classifier"] = 1 # Options for alternate fixing (number of step with fixed classifier)
    args_train["nb_step_fixed_selector"] = 1 # Options for alternate fixing (number of step with fixed selector)
    args_train["nb_step_all_free"] = 1 # Options for alternate fixing (number of step with all free)
    args_train["ratio_class_selection"] = 1.0 # Options for alternate ordinary Ratio of training with only classification compared to selection
    args_train["print_every"] = 1

    args_train["sampling_subset_size"] = 2 # Sampling size for the subset 
    args_train["use_cuda"] = torch.cuda.is_available()
    args_train["fix_classifier_parameters"] = False
    args_train["fix_selector_parameters"] = False
    args_train["post_hoc"] = False
    args_train["argmax_post_hoc"] = False
    args_train["post_hoc_guidance"] = None

    args_compiler = {}
    args_compiler["optim_classification"] = partial(Adam, lr=1e-4, weight_decay = 1e-3) #Learning rate for classification module
    args_compiler["optim_selection"] = partial(Adam, lr=1e-3, weight_decay = 1e-3) # Learning rate for selection module
    args_compiler["optim_selection_var"] = partial(Adam, lr=1e-4, weight_decay = 1e-3) # Learning rate for the variationnal selection module used in Variationnal Training
    args_compiler["optim_distribution_module"] = partial(Adam, lr=1e-4, weight_decay = 1e-3) # Learning rate for the feature extractor if any
    args_compiler["optim_baseline"] = partial(Adam, lr=1e-4, weight_decay = 1e-3) # Learning rate for the baseline network
    args_compiler["optim_autoencoder"] = partial(Adam, lr=1e-4, weight_decay = 1e-3)
    args_compiler["optim_post_hoc"] = partial(Adam, lr=1e-4, weight_decay = 1e-3)

    args_compiler["scheduler_classification"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9) #Learning rate for classification module
    args_compiler["scheduler_selection"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9) # Learning rate for selection module
    args_compiler["scheduler_selection_var"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9) # Learning rate for the variationnal selection module used in Variationnal Training
    args_compiler["scheduler_distribution_module"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9) # Learning rate for the feature extractor if any
    args_compiler["scheduler_baseline"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9) # Learning rate for the baseline network
    args_compiler["scheduler_autoencoder"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9)
    args_compiler["scheduler_post_hoc"] = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma = 0.9)
    
    args_test = {}
    args_test["nb_sample_z_mc_test"] = 1
    args_test["nb_sample_z_iwae_test"] = 1
    args_test["liste_mc"] = [(1,1,1,1), (100,1,1,1), (1,100,1,1), (1,1,100,1), (1,1,1,100)]

    return  args_output, args_dataset, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, args_classification_distribution_module
