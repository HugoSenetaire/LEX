
from .instantiate_utils import *
from .convert_args import convert_all
from .utils import load_model
import pickle as pkl

def instantiate(complete_args, dataset = None):
    """
    From the dictionnary of complete args, get the interpretable module instantiated with all networks set up
    """
    complete_args_converted = convert_all(complete_args)

    if complete_args_converted.args_classification.module_imputation_parameters is not None :
        if "path_module" in complete_args_converted.args_classification.module_imputation_parameters.keys() :
            path_module = complete_args_converted.args_classification.module_imputation_parameters["path_module"]
            parameters_path_module = os.path.join(os.path.join(path_module, "parameters"), "parameters.pkl")
            args_module = pkl.load(open(parameters_path_module, "rb"))
            interpretable_module, _ = instantiate(args_module)
            interpretable_module = load_full_module(path_module, interpretable_module)
            complete_args_converted.args_classification.module_imputation_parameters["module"] = interpretable_module
    imputation = get_imputation_method(complete_args_converted.args_classification, dataset,)

    ### Networks :
    classifier, selector, baseline, selector_var, reshape_mask_function = get_networks(complete_args_converted.args_classification,
                                                                        complete_args_converted.args_selection,
                                                                        complete_args_converted.args_trainer,
                                                                        complete_args_converted.args_interpretable_module,
                                                                        complete_args_converted.args_dataset,
                                                                        dataset=dataset)
    ### Regularization method :
    regularization = get_regularization_method(complete_args_converted.args_selection, complete_args_converted.args_distribution_module)
    
    ## Sampling :
    distribution_module = get_distribution_module_from_args(complete_args_converted.args_distribution_module)
    classification_distribution_module = get_distribution_module_from_args(complete_args_converted.args_classification_distribution_module)


    ### Imputation :


    ### Complete Module :
    if classifier is not None :
        prediction_module = PredictionModule(classifier, imputation=imputation, input_size=complete_args_converted.args_dataset.dataset_input_dim,)
    else :
        prediction_module = None
    if selector is not None:
        selection_module =  SelectionModule(selector,
                        activation=complete_args_converted.args_selection.activation,
                        regularization=regularization,
                        )
    else :
        selection_module = None
    selection_module_var = None
    interpretable_module = get_complete_module(complete_args_converted.args_interpretable_module.interpretable_module,
                                                prediction_module,
                                                selection_module,
                                                selection_module_var,
                                                distribution_module = distribution_module,
                                                classification_distribution_module = classification_distribution_module,
                                                reshape_mask_function = reshape_mask_function,
                                                dataset = dataset,
                                                args_selection=complete_args_converted.args_selection,
                                                )
    
    if complete_args_converted.args_train.use_cuda:
        interpretable_module.cuda()
        
    return interpretable_module, complete_args_converted

def load_full_module(final_path, interpretable_module, suffix = "_best"):
    """
    Load weights from a path
    """
    try :
        interpretable_module.prediction_module, _,_,_ = load_model(final_path,
                                    prediction_module=interpretable_module.prediction_module,
                                    suffix=suffix)
    except AttributeError:
        print("Interpretation module has no prediction module")

    try :
        _,interpretable_module.selection_module,_,_ = load_model(final_path,
                                    selection_module=interpretable_module.selection_module,
                                    suffix=suffix)
    except AttributeError:
        print("Interpretation module has no selection module")
    

    try :  
        _,_,interpretable_module.distribution_module,_ = load_model(final_path,
                                    distribution_module=interpretable_module.distribution_module,
                                    suffix=suffix)
    except AttributeError:
        print("Interpretation module has no distribution module")

    try :
        _,_,_,interpretable_module.baseline = load_model(final_path,
                                    selection_module=interpretable_module.baseline,
                                    suffix=suffix)
    except AttributeError:  
        print("Interpretation module has no baseline")


    return interpretable_module