import os
import sys
current_file_path = os.path.abspath(__file__)
while(not current_file_path.endswith("MissingDataTraining")):
    current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)


import missingDataTrainingModule
from experiment_class import complete_args
from datasets import *

import matplotlib.pyplot as plt
from torch.distributions import *
from torch.optim import *
import torch
from functools import partial

def get_default(args = None):
    if args is None :
        args = complete_args()

    args.args_output.path = os.path.join(os.path.dirname(missingDataTrainingModule.__path__[0]), "Experiments") # Path to results
    args.args_output.save_weights = True
    args.args_output.experiment_name = "REINFORCE"


    args.args_trainer.complete_trainer = missingDataTrainingModule.SELECTION_BASED_CLASSIFICATION
    args.args_trainer.monte_carlo_gradient_estimator = missingDataTrainingModule.PytorchDistributionUtils.gradientestimator.REINFORCE # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args.args_trainer.save_every_epoch = 1
    args.args_trainer.save_epoch_function = lambda epoch, nb_epoch: (epoch % args.args_trainer.save_every_epoch == 0) or (epoch == nb_epoch-1) or (epoch<10)
    args.args_trainer.baseline = None
    args.args_trainer.reshape_mask_function = missingDataTrainingModule.utils_reshape.CollapseInBatch


    args.args_dataset.dataset = DiagDataset
    args.args_dataset.loader = LoaderArtificial
    args.args_dataset.root_dir = os.path.join(args.args_output.path, "datasets")
    args.args_dataset.batch_size_train = 1000
    args.args_dataset.batch_size_test = 1000
    args.args_dataset.noise_function = None
    args.args_dataset.cov = torch.tensor(1.0, dtype=torch.float32)
    args.args_dataset.covariance_type = "spherical" #Choice spherical diagonal full
    args.args_dataset.mean = torch.tensor(0.0, dtype=torch.float32)
    args.args_dataset.download = True
    args.args_dataset.dim_input = 11
    args.args_dataset.used_dim = 10
    args.args_dataset.give_index = True



    args.args_classification.input_size_classification_module = (1,2) # Size before imputation
    args.args_classification.classifier = missingDataTrainingModule.Classification.classification_network.RealXClassifier
    args.args_classification.imputation = missingDataTrainingModule.Classification.imputation.ConstantImputation
    args.args_classification.cste_imputation = 0
    args.args_classification.sigma_noise_imputation = 1.0
    args.args_classification.add_mask = False
    args.args_classification.module_imputation = None
    args.args_classification.nb_imputation_iwae = 1
    args.args_classification.nb_imputation_iwae_test = None #If none is given, turn to 1
    args.args_classification.nb_imputation_mc = 1
    args.args_classification.nb_imputation_mc_test = None #If none is given, turn to 1
    args.args_classification.reconstruction_regularization = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args.args_classification.network_reconstruction = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args.args_classification.lambda_reconstruction = 0.01 # Parameter for controlling the reconstruction regularization
    args.args_classification.post_process_regularization = None # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args.args_classification.network_post_process = None # Autoencoder Network to use
    args.args_classification.post_process_trainable = False # If true, pretrain the autoencoder with the training data
    args.args_classification.mask_reg = None
    args.args_classification.mask_reg_rate = 0.5


    args.args_selection.input_size_selector = (1,2)
    args.args_selection.output_size_selector = (1,2)
    args.args_selection.selector = missingDataTrainingModule.Selection.selective_network.RealXSelector
    args.args_selection.selector_var = None 
    args.args_selection.activation = torch.nn.LogSigmoid()

    args.args_selection.trainable_regularisation = False
    args.args_selection.regularization = missingDataTrainingModule.Selection.regularization_module.LossRegularization
    args.args_selection.lambda_reg = 0.0 
    args.args_selection.rate = 0.0
    args.args_selection.loss_regularization = "L1" # L1, L2 
    args.args_selection.batched = False
    args.args_selection.regularization_var =  missingDataTrainingModule.Selection.regularization_module.LossRegularization
    args.args_selection.lambda_regularization_var = 0.0
    args.args_selection.rate_var = 0.1
    args.args_selection.loss_regularization_var = "L1"
    args.args_selection.batched_var = False



    args.args_distribution_module.distribution_module = missingDataTrainingModule.PytorchDistributionUtils.wrappers.DistributionModule
    args.args_distribution_module.distribution = Bernoulli
    args.args_distribution_module.distribution_relaxed = RelaxedBernoulli
    args.args_distribution_module.temperature_init = 1.0
    args.args_distribution_module.test_temperature = 1e-5
    args.args_distribution_module.scheduler_parameter = missingDataTrainingModule.PytorchDistributionUtils.wrappers.regular_scheduler
    args.args_distribution_module.antitheis_sampling = False 


    
    args.args_classification_distribution_module.distribution_module = missingDataTrainingModule.PytorchDistributionUtils.wrappers.FixedBernoulli
    args.args_classification_distribution_module.distribution = Bernoulli
    args.args_classification_distribution_module.distribution_relaxed = RelaxedBernoulli
    args.args_classification_distribution_module.temperature_init = 0.1
    args.args_classification_distribution_module.test_temperature = 1e-5
    args.args_classification_distribution_module.scheduler_parameter = missingDataTrainingModule.PytorchDistributionUtils.wrappers.regular_scheduler
    args.args_classification_distribution_module.antitheis_sampling = False 



    args.args_train.nb_epoch = 10 # Training the complete model
    args.args_train.nb_epoch_post_hoc = 0 # Training post_hoc
    args.args_train.nb_epoch_pretrain_selector = 0 # Pretrain selector
    args.args_train.use_regularization_pretrain_selector = False # Use regularization when pretraining the selector
    args.args_train.nb_epoch_pretrain = 0 # Training the complete model 
    args.args_train.nb_sample_z_train_monte_carlo = 1
    args.args_train.nb_sample_z_train_IWAE = 1  # Number K in the IWAE-similar loss
    args.args_train.loss_function = "NLL" # NLL, MSE

    args.args_train.training_type = "classic" # Options are .classic "alternate_ordinary", "alternate_fixing"]
    args.args_train.nb_step_fixed_classifier = 1 # Options for alternate fixing (number of step with fixed classifier)
    args.args_train.nb_step_fixed_selector = 1 # Options for alternate fixing (number of step with fixed selector)
    args.args_train.nb_step_all_free = 1 # Options for alternate fixing (number of step with all free)
    args.args_train.ratio_class_selection = 1.0 # Options for alternate ordinary Ratio of training with only classification compared to selection
    args.args_train.print_every = 1000

    args.args_train.sampling_subset_size = 2 # Sampling size for the subset 
    args.args_train.use_cuda = torch.cuda.is_available()
    args.args_train.fix_classifier_parameters = False
    args.args_train.fix_selector_parameters = False
    args.args_train.post_hoc = False
    args.args_train.argmax_post_hoc = False
    args.args_train.post_hoc_guidance = None

   
    args.args_compiler.optim_classification = "ADAM" #Learning rate for classification module
    args.args_compiler.optim_selection = "ADAM" # Learning rate for selection module
    args.args_compiler.optim_selection_var = "ADAM" # Learning rate for the variationnal selection module used in Variationnal Training
    args.args_compiler.optim_distribution_module = "ADAM" # Learning rate for the feature extractor if any
    args.args_compiler.optim_baseline = "ADAM" # Learning rate for the baseline network
    args.args_compiler.optim_autoencoder = "ADAM"
    args.args_compiler.optim_post_hoc = "ADAM"

    args.args_compiler.optim_classification_param = {"lr":1e-4,
                                                    "weight_decay" : 1e-3}  #Learning rate for classification module
    args.args_compiler.optim_selection_param = {"lr":1e-4,
                                                "weight_decay" : 1e-3}  # Learning rate for selection module
    args.args_compiler.optim_selection_var_param = {"lr":1e-4,
                                                    "weight_decay" : 1e-3}  # Learning rate for the variationnal selection module used in Variationnal Training
    args.args_compiler.optim_distribution_module_param = {"lr":1e-4,
                                                        "weight_decay" : 1e-3}  # Learning rate for the feature extractor if any
    args.args_compiler.optim_baseline_param = {"lr":1e-4,
                                                "weight_decay" : 1e-3}  # Learning rate for the baseline network
    args.args_compiler.optim_autoencoder_param = {"lr":1e-4,
                                                "weight_decay" : 1e-3} 
    args.args_compiler.optim_post_hoc_param = {"lr":1e-4,
                                                "weight_decay" : 1e-3} 




    args.args_compiler.scheduler_classification = "StepLR" #Learning rate for classification module
    args.args_compiler.scheduler_selection = "StepLR" # Learning rate for selection module
    args.args_compiler.scheduler_selection_var = "StepLR" # Learning rate for the variationnal selection module used in Variationnal Training
    args.args_compiler.scheduler_distribution_module = "StepLR" # Learning rate for the feature extractor if any
    args.args_compiler.scheduler_baseline = "StepLR" # Learning rate for the baseline network
    args.args_compiler.scheduler_autoencoder = "StepLR"
    args.args_compiler.scheduler_post_hoc = "StepLR"
    
    args.args_compiler.scheduler_classification_param = {"step_size": 1000,
                                                         "gamma": 0.9} #Learning rate for classification module
    args.args_compiler.scheduler_selection_param = {"step_size": 1000,
                                                         "gamma": 0.9} # Learning rate for selection module
    args.args_compiler.scheduler_selection_var_param = {"step_size": 1000,
                                                         "gamma": 0.9} # Learning rate for the variationnal selection module used in Variationnal Training
    args.args_compiler.scheduler_distribution_module_param = {"step_size": 1000,
                                                         "gamma": 0.9} # Learning rate for the feature extractor if any
    args.args_compiler.scheduler_baseline_param = {"step_size": 1000,
                                                         "gamma": 0.9} # Learning rate for the baseline network
    args.args_compiler.scheduler_autoencoder_param = {"step_size": 1000,
                                                         "gamma": 0.9}
    args.args_compiler.scheduler_post_hoc_param = {"step_size": 1000,
                                                         "gamma": 0.9}
    
    args.args_test.nb_sample_z_mc_test = 1
    args.args_test.nb_sample_z_iwae_test = 1
    args.args_test.liste_mc = [(1,1,1,1), (10,1,1,1), (1,10,1,1), (1,1,10,1), (1,1,1,10)]

    return  args
