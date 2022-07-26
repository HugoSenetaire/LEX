from .Prediction import *
from .PytorchDistributionUtils import *
from .Selection import *
from .classification_training import *
from .interpretation_training import *
from .selection_training import *



def get_imputation_method(args_classification, dataset):

    # Instantiate mask regularization method
    if args_classification.mask_reg is not None :
        mask_reg = args_classification.mask_reg(rate = args_classification.mask_reg_rate)
    else :
        mask_reg = None
    
    # Instantiate post process regularization method
    if args_classification.post_process_regularization is not None :
        post_process_regularization = args_classification.post_process_regularization(
                                        network_post_process = args_classification.network_post_process,
                                        trainable = args_classification.post_process_trainable,
                                        sigma_noise = args_classification.post_process_sigma_noise,
                                        )       
    else :
        post_process_regularization = None
    
    # Instantiate reconstruction regualrization method
    if args_classification.reconstruction_regularization is not None :
        reconstruction_regularization = args_classification.reconstruction_regularization(
                                        network_reconstruction = args_classification.network_reconstruction,
                                        lambda_reconstruction = args_classification.lambda_reconstruction,
                                        )
    else :
        reconstruction_regularization = None
    

    imputation = args_classification.imputation(
                    nb_imputation_mc = args_classification.nb_imputation_mc,
                    nb_imputation_mc_test = args_classification.nb_imputation_mc_test,
                    nb_imputation_iwae = args_classification.nb_imputation_iwae,
                    nb_imputation_iwae_test = args_classification.nb_imputation_iwae_test,
                    reconstruction_reg = reconstruction_regularization,
                    mask_reg = mask_reg,
                    post_process_regularization = post_process_regularization,
                    add_mask = args_classification.add_mask,
                    dataset = dataset,
                    module = args_classification.module_imputation,
                    cste =  args_classification.cste_imputation,
                    sigma = args_classification.sigma_noise_imputation,
                    )
    #TODO : That's a very poor way to handle multiple possible arguments, can lead to a lot of bugs, check that.

    return imputation



def get_loss_function(args_train, output_dim):
    if args_train.loss_function== "MSE" :
        loss_function = MSELossLastDim(reduction='none')
    elif args_train.loss_function== "NLL" :
        if output_dim == 1 :
            raise ValueError("NLL loss is not defined for a regression problem")
        if args_train.post_hoc and (not args_train.argmax_post_hoc):
            loss_function = continuous_NLLLoss(reduction='none')
        else :
            loss_function = NLLLossAugmented(reduction='none')
    else :
        raise ValueError("Unknown loss function") 
    
    return loss_function
        

def get_distribution_module_from_args(args_distribution_module):
    if args_distribution_module.distribution_module is None :
        return None
    dic_args_distribution_module = vars(args_distribution_module)
    distribution_module = args_distribution_module.distribution_module(**dic_args_distribution_module)
    return distribution_module

def get_optim(module, args_optimizer, args_optimizer_param, args_scheduler, args_scheduler_param):
    optimizer = None
    scheduler = None

    if args_optimizer is not None and module is not None and len(list(module.parameters())) > 0 :
        optimizer = args_optimizer(module.parameters(), **args_optimizer_param)
        if args_scheduler is not None :
            scheduler = args_scheduler(optimizer, **args_scheduler_param)

    return optimizer, scheduler

def get_networks(args_classification, args_selection, args_trainer, output_category):
    input_size_classifier = args_classification.input_size_classification_module
    input_size_baseline = args_classification.input_size_classification_module
    classifier =  args_classification.classifier(input_size_classifier, output_category)

    if args_trainer.baseline is not None :
        baseline = args_trainer.baseline(input_size_baseline, output_category)
    else :
        baseline = None

    input_size_selector = args_selection.input_size_selector
    output_size_selector = args_selection.output_size_selector

    try :
        kernel_size = args_selection.kernel_size
        kernel_stride = args_selection.kernel_stride
        output_size_selector = calculate_blocks_patch(input_size_selector, kernel_size, kernel_stride)
        args_selection.output_size_selector = output_size_selector
    except AttributeError:
        kernel_size = None
        kernel_stride = None
    
    try :
        reshape_mask_function = args_trainer.reshape_mask_function(input_size_classifier = input_size_classifier,
                                                                    output_size_selector = output_size_selector,
                                                                    kernel_size = kernel_size,
                                                                    kernel_stride = kernel_stride)
    except :
        reshape_mask_function = args_trainer.reshape_mask_function(size = input_size_classifier)

    try : 
        selector = args_selection.selector(input_size_selector, output_size_selector, kernel_size, kernel_stride)
    except TypeError:
        selector = args_selection.selector(input_size_selector, output_size_selector)
    

    if args_selection.selector_var is not None :
        try : 
            selector_var = args_selection.selector_var(input_size_selector, output_size_selector, kernel_size, kernel_stride)
        except TypeError :
            selector_var = args_selection.selector_var(input_size_selector, output_size_selector)
    else :
        selector_var = None

    return classifier, selector, baseline, selector_var, reshape_mask_function

def get_regularization_method(args_selection, args_distribution_module):
    """
    Get the regularization method for the selection module. 
    Note that if you used a self regularized distribution, 
    (ie distribution that already limits the number of element selected),
    no regularization method is explicitely used.
    
    """
    sampling_distrib = args_distribution_module.distribution
    if sampling_distrib in self_regularized_distributions :
        k = int(np.round(args_selection.rate * np.prod(args_selection.output_size_selector),))
        if k == 0 :
            print("Warning : k = 0, you need to select at least one variable. K =1 is used instead.")
            k = 1
        args_distribution_module.distribution = partial(sampling_distrib, k=k)
    
    dic_args_selection = vars(args_selection)
    regularization = args_selection.regularization(**dic_args_selection)
    return regularization

def check_parameters_compatibility(args_classification, args_selection, args_distribution_module, args_trainer, args_train, args_test, args_output):
    sampling_distrib = args_distribution_module.distribution
    activation = args_selection.activation
    if sampling_distrib in [RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2XDistribution_STE, L2XDistribution] \
        and activation is torch.nn.LogSoftmax() :
        raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")
    
def get_training_method(trainer, args_train, ordinaryTraining ):
    if args_train.training_type == "classic" :  # Options are .classic "alternate_ordinary", "alternate_fixing"]
        return trainer.classic_train_epoch
    elif args_train.training_type == "alternate_ordinary" :
        return partial(trainer.alternate_ordinary_train_epoch,
                ratio_class_selection = args_train.ratio_class_selection,
                ordinaryTraining = ordinaryTraining)
    elif args_train.training_type == "alternate_fixing" :
        return partial(trainer.alternate_fixing_train_epoch, 
                nb_step_fixed_classifier = args_train.nb_step_fixed_classifier,
                nb_step_fixed_selector = args_train.nb_step_fixed_selector,
                nb_step_all_free = args_train.nb_step_all_free,
                )
    else :
        raise ValueError(f"Training type {args_train['training_type']} is not implemented")
    
