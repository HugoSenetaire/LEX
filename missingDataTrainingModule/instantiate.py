from .Prediction import *
from .PytorchDistributionUtils import *
from .Selection import *
from .EvaluationUtils import *
from .InterpretableModel import *
from .utils import class_or_reg
from .Trainer import SINGLE_LOSS, SEPARATE_LOSS, trainingWithSelection, ordinaryPredictionTraining
from functools import partial

def get_imputation_method(args_classification, dataset,):

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



def get_loss_function(loss_function, args_train, output_dim):
    problem_type = class_or_reg(output_dim)
    
    if loss_function== "MSE" :
        if problem_type =="regression" :
            loss_function = MSE_Regression(reduction='none')
        else :
            loss_function = BrierScore(reduction='none')
    elif loss_function== "NLL" :
        if problem_type == "regression" :
            raise ValueError("NLL loss is not defined for a regression problem")
        if args_train.post_hoc and (not args_train.argmax_post_hoc):
            loss_function = continuous_NLLLoss(reduction='none')
        else :
            loss_function = NLLLossAugmented(reduction='none')
    elif loss_function == None :
        loss_function = None
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

def get_networks(args_classification, args_selection, args_trainer, args_interpretable_module, dataset):
    
    dim_output = dataset.get_dim_output()
    input_size_classifier = args_classification.input_size_prediction_module
    if args_classification.add_mask :
        init_shape = dataset.get_dim_input()
        input_size_classifier = torch.Size((init_shape[0]+1,)) + torch.Size((init_shape[1:]))
    input_size_baseline = args_classification.input_size_prediction_module
    if args_classification.classifier is DatasetBasedClassifier :
        classifier = args_classification.classifier(dataset)
    else :
        classifier =  args_classification.classifier(input_size_classifier, dim_output)

    if args_trainer.baseline is not None :
        baseline = args_trainer.baseline(input_size_baseline, dim_output)
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
        reshape_mask_function = args_interpretable_module.reshape_mask_function(input_size_classifier = args_classification.input_size_prediction_module,
                                                                    output_size_selector = output_size_selector,
                                                                    kernel_size = kernel_size,
                                                                    kernel_stride = kernel_stride)
    except :
        reshape_mask_function = args_interpretable_module.reshape_mask_function(size = args_classification.input_size_prediction_module)

    if args_selection.selector is None :
        selector = None
    else :
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

    if args_classification.network_reconstruction == "self" :
        args_classification.network_reconstruction = classifier

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
    
    if args_selection.regularization is None :
        return None
    else :
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
    

def get_complete_module(interpretable_module,
                        prediction_module,
                        selection_module,
                        selection_module_var,
                        distribution_module = None,
                        classification_distribution_module = None,
                        reshape_mask_function = None,
                        dataset = None,
                        args_selection = None,
                        ):

    if interpretable_module is COUPLED_SELECTION :
        interpretable_module = COUPLED_SELECTION(prediction_module,
                                selection_module,
                                distribution_module,
                                reshape_mask_function)
    elif interpretable_module is DECOUPLED_SELECTION :
        interpretable_module = DECOUPLED_SELECTION(prediction_module,
                        selection_module,
                        distribution_module,
                        classification_distribution_module,
                        reshape_mask_function)
    elif interpretable_module is PredictionCompleteModel :
        interpretable_module = PredictionCompleteModel(prediction_module,)      

    elif interpretable_module is trueSelectionCompleteModel :
        interpretable_module = trueSelectionCompleteModel(prediction_module, dataset)
    elif interpretable_module is EVAL_X :
        interpretable_module = EVAL_X(prediction_module,
                    classification_distribution_module,
                    reshape_mask_function,
                    mask_dimension=args_selection.output_size_selector,
        )                 
    else :
        raise ValueError(f"Interpretable module {interpretable_module} is not implemented")

    return interpretable_module

def get_trainer(trainer,
                interpretable_module,
                monte_carlo_gradient_estimator,
                baseline,
                fix_classifier_parameters,
                fix_selector_parameters,
                post_hoc_guidance,
                post_hoc,
                argmax_post_hoc,
                loss_function,
                loss_function_selection = None,
                nb_sample_z_monte_carlo = 1,
                nb_sample_z_iwae = 1,
                nb_sample_z_monte_carlo_classification = 1,
                nb_sample_z_iwae_classification = 1,
                ):
    
    if trainer is SINGLE_LOSS :
        trainer = SINGLE_LOSS(interpretable_module = interpretable_module,
                            monte_carlo_gradient_estimator= monte_carlo_gradient_estimator,
                            baseline = baseline,
                            fix_classifier_parameters = fix_classifier_parameters,
                            fix_selector_parameters = fix_selector_parameters,
                            post_hoc_guidance = post_hoc_guidance,
                            post_hoc = post_hoc,
                            argmax_post_hoc = argmax_post_hoc,
                            loss_function = loss_function,
                            nb_sample_z_monte_carlo = nb_sample_z_monte_carlo,
                            nb_sample_z_iwae = nb_sample_z_iwae)
    elif trainer is SEPARATE_LOSS :
        trainer = SEPARATE_LOSS(interpretable_module = interpretable_module,
                            monte_carlo_gradient_estimator= monte_carlo_gradient_estimator,
                            baseline = baseline,
                            fix_classifier_parameters = fix_classifier_parameters,
                            fix_selector_parameters = fix_selector_parameters,
                            post_hoc_guidance = post_hoc_guidance,
                            post_hoc = post_hoc,
                            argmax_post_hoc = argmax_post_hoc,
                            loss_function = loss_function,
                            loss_function_selection= loss_function_selection,
                            nb_sample_z_monte_carlo = nb_sample_z_monte_carlo,
                            nb_sample_z_iwae = nb_sample_z_iwae,
                            nb_sample_z_monte_carlo_classification = nb_sample_z_monte_carlo_classification,
                            nb_sample_z_iwae_classification = nb_sample_z_iwae_classification
                            )
                    
    elif trainer is ordinaryPredictionTraining :
        trainer = ordinaryPredictionTraining(interpretable_module = interpretable_module,
                    post_hoc = post_hoc,
                    post_hoc_guidance = post_hoc_guidance,
                    argmax_post_hoc = argmax_post_hoc,
                    loss_function = loss_function,
                    nb_sample_z_monte_carlo = nb_sample_z_monte_carlo,
                    nb_sample_z_iwae = nb_sample_z_iwae,
                    )
        
    elif trainer is trainingWithSelection :
        trainer = trainingWithSelection(interpretable_module = interpretable_module,
                    post_hoc = post_hoc,
                    post_hoc_guidance = post_hoc_guidance,
                    argmax_post_hoc = argmax_post_hoc,
                    loss_function = loss_function,
                    nb_sample_z_monte_carlo = nb_sample_z_monte_carlo,
                    nb_sample_z_iwae = nb_sample_z_iwae,)
    else :
        raise ValueError(f"Trainer {trainer} is not implemented")

    return trainer

def compile_trainer(
    trainer,
    trainer_type,
    optim_classification,
    optim_selection = None,
    scheduler_classification = None,
    scheduler_selection = None,
    optim_baseline = None,
    scheduler_baseline = None,
    optim_distribution_module = None,
    scheduler_distribution_module = None,
) :

    if trainer_type is SINGLE_LOSS or trainer_type is SEPARATE_LOSS :
        trainer.compile(
            optim_classification = optim_classification,
            optim_selection = optim_selection,
            scheduler_classification = scheduler_classification,
            scheduler_selection = scheduler_selection,
            optim_baseline = optim_baseline,
            scheduler_baseline = scheduler_baseline,
            optim_distribution_module = optim_distribution_module,
            scheduler_distribution_module = scheduler_distribution_module,
            )
    
    elif trainer_type is ordinaryPredictionTraining or trainer_type is trainingWithSelection :
        trainer.compile(
            optim_classification = optim_classification,
            scheduler_classification = scheduler_classification,
            )
    else :
        raise ValueError(f"Trainer {trainer} is not implemented")

    return trainer