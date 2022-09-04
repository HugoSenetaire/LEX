from .Prediction import get_pred_network, dic_reconstruction_regularization, get_imputation_type, get_post_process_reg, get_maskreg
from .PytorchDistributionUtils import *
from .Selection import *
from .InterpretableModel import *
from .Trainer import *
import copy

def convert_trainer(args_trainer,):
    args_trainer.complete_trainer = get_complete_trainer(args_trainer.complete_trainer)

    if args_trainer.monte_carlo_gradient_estimator == "REBAR": 
        args_trainer.monte_carlo_gradient_estimator = PytorchDistributionUtils.gradientestimator.REBAR # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    elif args_trainer.monte_carlo_gradient_estimator == "PATHWISE":
        args_trainer.monte_carlo_gradient_estimator = PytorchDistributionUtils.gradientestimator.PathWise
    elif args_trainer.monte_carlo_gradient_estimator == "ALLCOMBINATION":
        args_trainer.monte_carlo_gradient_estimator = PytorchDistributionUtils.gradientestimator.AllCombination
    elif args_trainer.monte_carlo_gradient_estimator == "REINFORCE":
        args_trainer.monte_carlo_gradient_estimator = PytorchDistributionUtils.gradientestimator.REINFORCE
    else :
        raise ValueError(f"Unknown Gradient estimator {args_trainer.monte_carlo_gradient_estimator}")

    args_trainer.save_epoch_function = lambda epoch, nb_epoch: (epoch % args_trainer.save_every_epoch == 0) or (epoch == nb_epoch-1) or (epoch<10)
    args_trainer.baseline = get_pred_network(args_trainer.baseline,) # Write the get network function ? But then I also need some information about the classification part.
    
    return args_trainer

def convert_interpretable_module(args_interpretable_module,):
    args_interpretable_module.interpretable_module = get_interpretable_module(args_interpretable_module.interpretable_module)
    args_interpretable_module.reshape_mask_function = get_reshape_mask(args_interpretable_module.reshape_mask_function)
    return args_interpretable_module

def convert_classification(args_classification):
    args_classification.classifier = get_pred_network(args_classification.classifier) 
    args_classification.imputation = get_imputation_type(args_classification.imputation)


    args_classification.reconstruction_regularization = dic_reconstruction_regularization[args_classification.reconstruction_regularization] # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_classification.network_reconstruction = args_classification.network_reconstruction # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    
    
    args_classification.post_process_regularization = get_post_process_reg(args_classification.post_process_regularization) # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_classification.network_post_process = None # Autoencoder Network to use
    args_classification.post_process_trainable = False # If true, pretrain the autoencoder with the training data
    
    args_classification.mask_reg = get_maskreg(args_classification.mask_reg)
    args_classification.mask_reg_rate = 0.5

    return args_classification

def convert_selection(args_selection):
    
    args_selection.selector = get_selection_network(args_selection.selector) 
    args_selection.selector_var = get_selection_network(args_selection.selector_var) 

    if args_selection.activation == "LogSigmoid":
        args_selection.activation = torch.nn.LogSigmoid()
    elif args_selection.activation == "LogSoftmax":
        args_selection.activation = torch.nn.LogSoftmax(dim =-1)

    # For regularization :
    if args_selection.regularization == "LossRegularization" :
        args_selection.regularization = LossRegularization
    elif args_selection.regularization == "SoftmaxRegularization":
        args_selection.regularization = SoftmaxRegularization
    elif args_selection.regularization == "TopkRegularization" :
        args_selection.regularization == TopKRegularization
    elif args_selection.regularization == "None" or args_selection.regularization == None :
        args_selection.regularization = None
    else :
        raise ValueError("Unknown Regularization")

    if args_selection.regularization_var == None or args_selection.regularization_var == "None" :
        args_selection.regularization_var = None
    elif args_selection.regularization_var == "LossRegularization" :
        args_selection.regularization_var = LossRegularization
    elif args_selection.regularization_var == "SoftmaxRegularization":
        args_selection.regularization_var = SoftmaxRegularization
    elif args_selection.regularization_var == "TopkRegularization" :
        args_selection.regularization_var == TopKRegularization
    elif args_selection.regularization_var == "None" or args_selection.regularization_var == None :
        args_selection.regularization_var = None
    else :
        raise ValueError("Unknown Regularization")

    return args_selection

def convert_distribution(args_distribution):

    if args_distribution.distribution_module == "REBARBernoulli_STE":
        args_distribution.distribution_module = PytorchDistributionUtils.wrappers.REBARBernoulli_STE
    elif args_distribution.distribution_module == "REBARBernoulli" :
        args_distribution.distribution_module = PytorchDistributionUtils.wrappers.REBARBernoulli
    elif args_distribution.distribution_module == "DistributionWithTemperature" or args_distribution.distribution_module == "DistributionWithTemperatureParameter" :
        args_distribution.distribution_module = PytorchDistributionUtils.wrappers.DistributionWithTemperatureParameter
    elif args_distribution.distribution_module == "DistributionModule" :
        args_distribution.distribution_module = PytorchDistributionUtils.wrappers.DistributionModule
    elif args_distribution.distribution_module == "FixedBernoulli" :
        args_distribution.distribution_module = PytorchDistributionUtils.wrappers.FixedBernoulli
    elif args_distribution.distribution_module is None or args_distribution.distribution_module == "None" :
        args_distribution.distribution_module = None
    else :
        raise ValueError("Unknown distribution module")

    if args_distribution.distribution == "Bernoulli":
        args_distribution.distribution = torch.distributions.Bernoulli
    elif args_distribution.distribution == "RelaxedBernoulli" :
        args_distribution.distribution = torch.distributions.RelaxedBernoulli
    elif args_distribution.distribution == "L2XDistribution" :
        args_distribution.distribution = L2XDistribution
    elif args_distribution.distribution == "L2XDistribution_STE" :
        args_distribution.distribution = L2XDistribution_STE
    elif args_distribution.distribution == "RelaxedBernoulli_thresholded_STE" :
        args_distribution.distribution = RelaxedBernoulli_thresholded_STE
    elif args_distribution.distribution == "RelaxedSubsetSampling":
        args_distribution.distribution = RelaxedSubsetSampling
    elif args_distribution.distribution == "RelaxedSubsetSampling_STE":
        args_distribution.distribution = RelaxedSubsetSampling_STE
    elif args_distribution.distribution is None or args_distribution.distribution == "None" :
        args_distribution.distribution = None
    else :
        raise ValueError("Distribution not found")
    
    if args_distribution.distribution_relaxed == "Bernoulli":
        args_distribution.distribution_relaxed = torch.distributions.Bernoulli
    elif args_distribution.distribution_relaxed == "RelaxedBernoulli" :
        args_distribution.distribution_relaxed = torch.distributions.RelaxedBernoulli
    elif args_distribution.distribution_relaxed == "L2XDistribution" :
        args_distribution.distribution_relaxed = L2XDistribution
    elif args_distribution.distribution_relaxed == "L2XDistribution_STE" :
        args_distribution.distribution_relaxed = L2XDistribution_STE
    elif args_distribution.distribution_relaxed == "RelaxedBernoulli_thresholded_STE" :
        args_distribution.distribution_relaxed = RelaxedBernoulli_thresholded_STE
    elif args_distribution.distribution_relaxed == "RelaxedSubsetSampling":
        args_distribution.distribution_relaxed = RelaxedSubsetSampling
    elif args_distribution.distribution_relaxed == "RelaxedSubsetSampling_STE":
        args_distribution.distribution_relaxed = RelaxedSubsetSampling_STE
    elif args_distribution.distribution_relaxed is None or args_distribution.distribution_relaxed == "None" :
        args_distribution.distribution_relaxed = None
    else :
        raise ValueError("Distribution relaxed not found")


    if args_distribution.scheduler_parameter == "None" or args_distribution.scheduler_parameter == None :
        args_distribution.scheduler_parameter = None
    elif args_distribution.scheduler_parameter == "regular_scheduler" :
        args_distribution.scheduler_parameter = PytorchDistributionUtils.wrappers.regular_scheduler
    else :
        raise ValueError("Scheduler distribution not found")
    
    return args_distribution

def convert_optim(optim,):
    if optim == "ADAM":
        return torch.optim.Adam
    elif optim == "SGD" :
        return torch.optim.SGD
    else :
        raise ValueError("Distribution not found")


def convert_scheduler(scheduler,) :
    if scheduler == None or scheduler == "None" :
        return None
    elif scheduler == "StepLR" :
        return torch.optim.lr_scheduler.StepLR
    else :
        raise ValueError("Scheduler")
    
def convert_compiler(args_compiler):
    args_compiler.optim_classification = convert_optim(args_compiler.optim_classification)  #Learning rate for classification module
    args_compiler.optim_selection = convert_optim(args_compiler.optim_selection)  # Learning rate for selection module
    args_compiler.optim_selection_var = convert_optim(args_compiler.optim_selection_var)  # Learning rate for the variationnal selection module used in Variationnal Training
    args_compiler.optim_distribution_module = convert_optim(args_compiler.optim_distribution_module)  # Learning rate for the feature extractor if any
    args_compiler.optim_baseline = convert_optim(args_compiler.optim_baseline)  # Learning rate for the baseline network
    args_compiler.optim_autoencoder = convert_optim(args_compiler.optim_autoencoder) 
    args_compiler.optim_post_hoc = convert_optim(args_compiler.optim_post_hoc) 

    args_compiler.scheduler_classification = convert_scheduler(args_compiler.scheduler_classification)  #Learning rate for classification module
    args_compiler.scheduler_selection = convert_scheduler(args_compiler.scheduler_selection)  # Learning rate for selection module
    args_compiler.scheduler_selection_var = convert_scheduler(args_compiler.scheduler_selection_var)  # Learning rate for the variationnal selection module used in Variationnal Training
    args_compiler.scheduler_distribution_module = convert_scheduler(args_compiler.scheduler_distribution_module)  # Learning rate for the feature extractor if any
    args_compiler.scheduler_baseline = convert_scheduler(args_compiler.scheduler_baseline)  # Learning rate for the baseline network
    args_compiler.scheduler_autoencoder = convert_scheduler(args_compiler.scheduler_autoencoder) 
    args_compiler.scheduler_post_hoc = convert_scheduler(args_compiler.scheduler_post_hoc) 

    return args_compiler


def convert_all(complete_args):
    complete_args_converted = copy.deepcopy(complete_args)
    complete_args_converted.args_trainer = convert_trainer(complete_args_converted.args_trainer)
    complete_args_converted.args_distribution_module = convert_distribution(complete_args_converted.args_distribution_module)
    complete_args_converted.args_classification_distribution_module = convert_distribution(complete_args_converted.args_classification_distribution_module)
    complete_args_converted.args_selection = convert_selection(complete_args_converted.args_selection)
    complete_args_converted.args_classification = convert_classification(complete_args_converted.args_classification)
    complete_args_converted.args_compiler = convert_compiler(complete_args_converted.args_compiler)
    complete_args_converted.args_interpretable_module = convert_interpretable_module(complete_args_converted.args_interpretable_module)
    return complete_args_converted
