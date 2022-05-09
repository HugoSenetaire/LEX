import os
import torch
import torch.nn as nn

from missingDataTrainingModule.utils.loss import NLLLossAugmented

from .classification_training import ordinaryTraining, EVAL_X, trueSelectionTraining
from .interpretation_training import SELECTION_BASED_CLASSIFICATION, REALX
from .selection_training import selectionTraining
from .utils import MSELossLastDim, define_target, continuous_NLLLoss, fill_dic, save_dic, save_model
from .PytorchDistributionUtils.distribution import self_regularized_distributions, RelaxedSubsetSampling, RelaxedBernoulli_thresholded_STE, RelaxedSubsetSampling_STE, L2XDistribution_STE, L2XDistribution
from .Selection.selection_module import SelectionModule, calculate_blocks_patch
from .Classification.classification_module import ClassificationModule

from torch.distributions import *
from torch.optim import *
import numpy as np
from functools import partial


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def save_parameters(path, args_classification, args_selection, args_distribution, args_complete_trainer, args_train, args_test, args_output, args_compiler, args_classification_distribution, args_dataset = None):
    complete_path = os.path.join(path, "parameters")
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)


    with open(os.path.join(complete_path,"classification.txt"), "w") as f:
        f.write(str(args_classification))
    
    with open(os.path.join(complete_path,"selection.txt"), "w") as f:
        f.write(str(args_selection))

    with open(os.path.join(complete_path, "distribution_module.txt"), "w") as f:
        f.write(str(args_distribution))

    with open(os.path.join(complete_path,"complete_trainer.txt"), "w") as f:
        f.write(str(args_complete_trainer))

    with open(os.path.join(complete_path,"train.txt"), "w") as f:
        f.write(str(args_train))

    with open(os.path.join(complete_path,"test.txt"), "w") as f:
        f.write(str(args_test))

    with open(os.path.join(complete_path,"output.txt"), "w") as f:
        f.write(str(args_output))

    with open(os.path.join(complete_path,"compiler.txt"), "w") as f:
        f.write(str(args_compiler))

    with open(os.path.join(complete_path, "classification_distribution_module.txt"), "w") as f:
        f.write(str(args_classification_distribution))

    if args_dataset is not None :
        with open(os.path.join(complete_path, "dataset.txt"), "w") as f:
            f.write(str(args_dataset))


def get_imputation_method(args_classification, dataset):
    if args_classification["mask_reg"] is not None :
        mask_reg = args_classification["mask_reg"](rate = args_classification["mask_reg_rate"])
    else :
        mask_reg = None
    
    if args_classification["post_process_regularization"] is not None :
        post_process_regularization = args_classification["post_process_regularization"](
                                        network_post_process = args_classification["network_post_process"],
                                        trainable = args_classification["post_process_trainable"],
                                        sigma_noise = args_classification["post_process_sigma_noise"],
                                        )       
    else :
        post_process_regularization = None
    
    if args_classification["reconstruction_regularization"] is not None :
        reconstruction_regularization = args_classification["reconstruction_regularization"](
                                        network_reconstruction = args_classification["network_reconstruction"],
                                        lambda_reconstruction = args_classification["lambda_reconstruction"],
                                        )
    else :
        reconstruction_regularization = None
    

    imputation = args_classification["imputation"](
                                        nb_imputation_mc = args_classification["nb_imputation_mc"],
                                        nb_imputation_mc_test = args_classification["nb_imputation_mc_test"],
                                        nb_imputation_iwae = args_classification["nb_imputation_iwae"],
                                        nb_imputation_iwae_test = args_classification["nb_imputation_iwae_test"],
                                        reconstruction_reg = reconstruction_regularization,
                                        mask_reg = mask_reg,
                                        post_process_regularization = post_process_regularization,
                                        add_mask = args_classification["add_mask"],
                                        dataset = dataset,
                                        module = args_classification["module_imputation"],
                                        cste =  args_classification["cste_imputation"],
                                        sigma = args_classification["sigma_noise_imputation"],
                                        )
    #TODO : That's a very poor way to handle multiple possible arguments, can lead to a lot of bugs, check that.

    return imputation



def get_loss_function(args_train, output_dim):
    if args_train["loss_function"]== "MSE" :
        loss_function = MSELossLastDim(reduction='none')
    elif args_train["loss_function"]== "NLL" :
        if output_dim == 1 :
            raise ValueError("NLL loss is not defined for a regression problem")
        if args_train["post_hoc"] and (not args_train["argmax_post_hoc"]):
            loss_function = continuous_NLLLoss(reduction='none')
        else :
            loss_function = NLLLossAugmented(reduction='none')
    else :
        raise ValueError("Unknown loss function") 
    
    return loss_function
        

def get_distribution_module_from_args(args_distribution_module):
    if args_distribution_module["distribution_module"] is None :
        return None
    distribution_module = args_distribution_module["distribution_module"](**args_distribution_module)
    return distribution_module

def get_optim(module, args_optimizer, args_scheduler):
    optimizer = None
    scheduler = None

    if args_optimizer is not None and module is not None and len(list(module.parameters())) > 0 :
        optimizer = args_optimizer(module.parameters(),)
        if args_scheduler is not None :
            scheduler = args_scheduler(optimizer,)

    return optimizer, scheduler

def get_networks(args_classification, args_selection, args_complete_trainer, output_category):
    input_size_classifier = args_classification["input_size_classification_module"]
    input_size_baseline = args_classification["input_size_classification_module"]
    classifier =  args_classification["classifier"](input_size_classifier, output_category)

    if args_complete_trainer["baseline"] is not None :
        baseline = args_complete_trainer["baseline"](input_size_baseline, output_category)
    else :
        baseline = None

    input_size_selector = args_selection["input_size_selector"]
    output_size_selector = args_selection["output_size_selector"]

    try :
        kernel_size = args_selection["kernel_size"]
        kernel_stride = args_selection["kernel_stride"]
        output_size_selector = calculate_blocks_patch(input_size_selector, kernel_size, kernel_stride)
        args_selection["output_size_selector"] = output_size_selector
    except KeyError:
        kernel_size = None
        kernel_stride = None
    
    try :
        reshape_mask_function = args_complete_trainer["reshape_mask_function"](input_size_classifier = input_size_classifier,
                                                                    output_size_selector = output_size_selector,
                                                                    kernel_size = kernel_size,
                                                                    kernel_stride = kernel_stride)
    except :
        reshape_mask_function = args_complete_trainer["reshape_mask_function"](size = input_size_classifier)

    try : 
        selector = args_selection["selector"](input_size_selector, output_size_selector, kernel_size, kernel_stride)
    except TypeError:
        selector = args_selection["selector"](input_size_selector, output_size_selector)
    

    if args_selection["selector_var"] is not None :
        try : 
            selector_var = args_selection["selector_var"](input_size_selector, output_size_selector, kernel_size, kernel_stride)
        except TypeError :
            selector_var = args_selection["selector_var"](input_size_selector, output_size_selector)
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
    sampling_distrib = args_distribution_module["distribution"]
    if sampling_distrib in self_regularized_distributions :
        
        k = int(np.round(args_selection["rate"] * np.prod(args_selection["output_size_selector"]),))
        if k == 0 :
            print("Warning : k = 0, you need to select at least one variable. K =1 is used instead.")
            k = 1
        args_distribution_module["distribution"] = partial(sampling_distrib, k=k)
    
    regularization = args_selection["regularization"](**args_selection)
    return regularization

def check_parameters_compatibility(args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_output):
    sampling_distrib = args_distribution_module["distribution"]
    activation = args_selection["activation"]
    if sampling_distrib in [RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2XDistribution_STE, L2XDistribution] \
        and activation is torch.nn.LogSoftmax() :
        raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")
    
def get_training_method(trainer, args_train, ordinaryTraining ):
    if args_train["training_type"] == "classic" :  # Options are ["classic", "alternate_ordinary", "alternate_fixing"]
        return trainer.classic_train_epoch
    elif args_train["training_type"] == "alternate_ordinary" :
        return partial(trainer.alternate_ordinary_train_epoch, ratio_class_selection = args_train["ratio_class_selection"], ordinaryTraining = ordinaryTraining)
    elif args_train["training_type"] == "alternate_fixing" :
        return partial(trainer.alternate_fixing_train_epoch, 
                nb_step_fixed_classifier = args_train["nb_step_fixed_classifier"],
                nb_step_fixed_selector = args_train["nb_step_fixed_selector"],
                nb_step_all_free = args_train["nb_step_all_free"],
                )
    else :
        raise ValueError(f"Training type {args_train['training_type']} is not implemented")
    


def experiment(dataset, loader, args_output, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, args_classification_distribution_module, name_modification = False, args_dataset = None):
    torch.random.manual_seed(0)
    dic_list = {}

    ### Prepare output path :
    origin_path = args_output["path"]
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)

    if name_modification :
        folder = os.path.join(origin_path,str(dataset))
        if not os.path.exists(folder):
            os.makedirs(folder)

        experiment_name = args_output["experiment_name"]
        final_path = os.path.join(folder, experiment_name)
    else :
        final_path = origin_path

    check_parameters_compatibility(args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_output)


    print("====================================================================================================================================================")
    print(f"Save at {final_path}")
    print(f"Dir at {os.path.dirname(final_path)}")
    print("====================================================================================================================================================")





    ### Regularization method :
    regularization = get_regularization_method(args_selection, args_distribution_module)


    ## Sampling :
    distribution_module = get_distribution_module_from_args(args_distribution_module)
    classification_distribution_module = get_distribution_module_from_args(args_classification_distribution_module)
    

    ### Imputation :
    imputation = get_imputation_method(args_classification, dataset)



    ### Networks :
    classifier, selector, baseline, selector_var, reshape_mask_function = get_networks(args_classification, args_selection, args_complete_trainer, dataset.get_dim_output())

    ### Loss Function :
    loss_function = get_loss_function(args_train, dataset.get_dim_output())

    
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    save_parameters(final_path, args_classification = args_classification,
                    args_selection = args_selection,
                    args_distribution= args_distribution_module,
                    args_complete_trainer= args_complete_trainer,
                    args_train = args_train,
                    args_test = args_test,
                    args_output=args_output,
                    args_compiler=args_compiler,
                    args_classification_distribution=args_classification_distribution_module,
                    args_dataset=args_dataset,
                    )
    

    ##### ============ Training POST-HOC ============= ####

    if args_train["post_hoc"] and args_train["post_hoc_guidance"] is not None :
        print("Training post-hoc guidance")
        post_hoc_classifier =  args_train["post_hoc_guidance"](args_selection["input_size_selector"], dataset.get_dim_output())
        post_hoc_guidance = ClassificationModule(post_hoc_classifier, imputation = imputation)



        trainer = ordinaryTraining(classification_module = post_hoc_guidance,)
        if args_train["use_cuda"]:
            trainer.cuda()

        optim_post_hoc, scheduler_post_hoc = get_optim(post_hoc_guidance, args_compiler["optim_post_hoc"], args_compiler["scheduler_post_hoc"])
        optim_post_hoc = args_compiler["optim_post_hoc"](post_hoc_guidance.parameters())
        trainer.compile(optim_classification=optim_post_hoc, scheduler_classification = scheduler_post_hoc)

        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(args_train["nb_epoch_post_hoc"]):
            dic_train = trainer.train_epoch(epoch, loader, loss_function=loss_function, save_dic = True, print_dic_bool= ((epoch+1) % args_train["print_every"] == 0),)
            total_dic_train = fill_dic(total_dic_train, dic_train)

            test_this_epoch = args_complete_trainer["save_epoch_function"](epoch, args_train["nb_epoch_post_hoc"])
            if test_this_epoch :
                dic_test = trainer.test(epoch, loader, )
                total_dic_test = fill_dic(total_dic_test, dic_test)

            
            
        save_dic(os.path.join(final_path,"train_post_hoc"), total_dic_train)
        save_dic(os.path.join(final_path,"test_post_hoc"), total_dic_test)

        dic_list["train_post_hoc"] = total_dic_train
        dic_list["test_post_hoc"]  = total_dic_test
        post_hoc_guidance.eval()
    else :
        post_hoc_guidance = None

    ##### ============ Modules initialisation for ordinary training ============:

    trainer_ordinary = None
    if (args_complete_trainer["complete_trainer"] is EVAL_X) or (args_complete_trainer["complete_trainer"] is REALX) :

            classification_module = ClassificationModule(classifier, imputation)

            if  args_complete_trainer["complete_trainer"] is REALX and args_train["nb_epoch_pretrain"] > 0 :
                post_hoc_guidance_eval_x = None
                post_hoc_eval_x = False
            else :
                post_hoc_guidance_eval_x = post_hoc_guidance 
                post_hoc_eval_x = args_train["post_hoc"]


            trainer_ordinary = EVAL_X(classification_module, 
                            reshape_mask_function = reshape_mask_function,
                            post_hoc_guidance = post_hoc_guidance_eval_x,
                            post_hoc = post_hoc_eval_x,
                            argmax_post_hoc = args_train["argmax_post_hoc"],
                            )
            if args_train["use_cuda"]:
                trainer_ordinary.cuda()

            optim_classification, scheduler_classification = get_optim(classification_module, args_compiler["optim_classification"], args_compiler["scheduler_classification"])
            trainer_ordinary.compile(optim_classification=optim_classification, scheduler_classification = scheduler_classification,)
            
            if args_complete_trainer["complete_trainer"] is EVAL_X :
                nb_epoch = args_train["nb_epoch"]
            else :
                nb_epoch = args_train["nb_epoch_pretrain"]
            total_dic_train = {}
            total_dic_test = {}
            for epoch in range(nb_epoch):
                dic_train = trainer_ordinary.train_epoch(epoch, loader, nb_sample_z_monte_carlo = args_train["nb_sample_z_train_monte_carlo"], loss_function = loss_function, save_dic = True, verbose=True)
                total_dic_train = fill_dic(total_dic_train, dic_train)
                test_this_epoch = args_complete_trainer["save_epoch_function"](epoch, nb_epoch)
                if test_this_epoch :
                    dic_test = trainer_ordinary.test(epoch, loader, liste_mc = args_test["liste_mc"])
                    total_dic_test = fill_dic(total_dic_test, dic_test)


            dic_list["train_pretraining_eval_x"] = total_dic_train
            dic_list["test_pretraining_eval_x"]  = total_dic_test

            if args_complete_trainer["complete_trainer"] is EVAL_X:
                return final_path, trainer_ordinary, loader, dic_list

    else :
        if args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trueSelectionTraining :
            nb_epoch = int(args_train["nb_epoch"])
            post_hoc_guidance_ordinary = post_hoc_guidance
            post_hoc_ordinary = args_train["post_hoc"]
            
        else :
            nb_epoch = args_train["nb_epoch_pretrain"]
            post_hoc_guidance_ordinary = None
            post_hoc_ordinary = False


        vanilla_classification_module = ClassificationModule(classifier,  imputation = imputation)

        if args_complete_trainer["complete_trainer"] is trueSelectionTraining :
            trainer_ordinary = trueSelectionTraining(vanilla_classification_module, 
                                    post_hoc_guidance = post_hoc_guidance_ordinary,
                                    post_hoc = post_hoc_ordinary,
                                    argmax_post_hoc = args_train["argmax_post_hoc"],
                                    )
        else :
            trainer_ordinary = ordinaryTraining(vanilla_classification_module,
                                    post_hoc_guidance = post_hoc_guidance_ordinary,
                                    post_hoc = post_hoc_ordinary,
                                    argmax_post_hoc = args_train["argmax_post_hoc"],
                                )

        if args_train["use_cuda"]:
            trainer_ordinary.cuda()

        optim_classification, scheduler_classification = get_optim(vanilla_classification_module, args_compiler["optim_classification"], args_compiler["scheduler_classification"])
        trainer_ordinary.compile(optim_classification=optim_classification, scheduler_classification = scheduler_classification,)


        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(int(nb_epoch)):
            dic_train = trainer_ordinary.train_epoch(epoch, loader, loss_function = loss_function, save_dic = True, verbose=True)
            total_dic_train = fill_dic(total_dic_train, dic_train)
            test_this_epoch = args_complete_trainer["save_epoch_function"](epoch, nb_epoch)
            if test_this_epoch :
                dic_test = trainer_ordinary.test(epoch, loader, liste_mc = args_test["liste_mc"])
                total_dic_test = fill_dic(total_dic_test, dic_test)


        if args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trueSelectionTraining :
            dic_list["train"] = total_dic_train
            dic_list["test"]  = total_dic_test
            save_dic(os.path.join(final_path,"train"), total_dic_train)
            save_dic(os.path.join(final_path,"test"), total_dic_test)
            return final_path, trainer_ordinary, loader, dic_list

        else :
            dic_list["train_pretraining"] = total_dic_train
            dic_list["test_pretaining"]  = total_dic_test
            
    ##### ======================= Training in selection ==========================:

    selection_module = SelectionModule(selector,
                    activation=args_selection["activation"],
                    regularization=regularization,
                    )

    if args_train["nb_epoch_pretrain_selector"] > 0 :
        selection_trainer = selectionTraining(selection_module, args_train["use_regularization_pretrain_selector"])
        optim_selection, scheduler_selection = get_optim(selection_module, args_compiler["optim_selection"], args_compiler["scheduler_selection"])
        selection_trainer.compile(optim_selection=optim_selection, scheduler_selection = scheduler_selection,)
        nb_epoch = args_train["nb_epoch_pretrain_selector"]
    
        if args_train["use_cuda"]:
            selection_trainer.cuda()

        if loader.dataset.optimal_S_train is None or loader.dataset.optimal_S_test is None :
            raise AttributeError("optimal_S_train or optimal_S_test not define for this dataset")

        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(int(nb_epoch)):
            dic_train = selection_trainer.train_epoch(epoch, loader, save_dic = True,)
            total_dic_train = fill_dic(total_dic_train, dic_train)

            test_this_epoch = args_complete_trainer["save_epoch_function"](epoch, nb_epoch)
            if test_this_epoch :
                dic_test = selection_trainer.test(epoch, loader, )
                total_dic_test = fill_dic(total_dic_test, dic_test)

        
            
        dic_list["train_selection_pretraining"] = total_dic_train
        dic_list["test_selection_pretaining"]  = total_dic_test


                


    ##### ============  Modules initialisation for complete training ===========:
   


    classification_module = ClassificationModule(classifier, imputation=imputation)
    trainer = args_complete_trainer["complete_trainer"](
        classification_module,
        selection_module,
        monte_carlo_gradient_estimator = args_complete_trainer["monte_carlo_gradient_estimator"],
        baseline = baseline,
        distribution_module = distribution_module,
        classification_distribution_module = classification_distribution_module,
        reshape_mask_function = reshape_mask_function,
        fix_classifier_parameters = args_train["fix_classifier_parameters"],
        fix_selector_parameters = args_train["fix_selector_parameters"],
        post_hoc_guidance = post_hoc_guidance,
        post_hoc = args_train["post_hoc"],
        argmax_post_hoc = args_train["argmax_post_hoc"],
        )

    if args_train["use_cuda"]:
        trainer.cuda()

    ####Optim_optim_classification :

    optim_classification, scheduler_classification = get_optim(classification_module, args_compiler["optim_classification"], args_compiler["scheduler_classification"]) 
    optim_selection, scheduler_selection = get_optim(selection_module, args_compiler["optim_selection"], args_compiler["scheduler_selection"])
    optim_baseline, scheduler_baseline = get_optim(baseline, args_compiler["optim_baseline"], args_compiler["scheduler_baseline"])
    optim_distribution_module, scheduler_distribution_module = get_optim(distribution_module, args_compiler["optim_distribution_module"], args_compiler["scheduler_distribution_module"])

    trainer.compile(optim_classification = optim_classification,
        optim_selection = optim_selection,
        scheduler_classification = scheduler_classification,
        scheduler_selection = scheduler_selection,
        optim_baseline = optim_baseline,
        scheduler_baseline = scheduler_baseline,
        optim_distribution_module = optim_distribution_module,
        scheduler_distribution_module = scheduler_distribution_module,
        )



###### Main module training :

    best_train_loss_in_test = float("inf")
    total_dic_train = {}
    total_dic_test = {}
    for epoch in range(args_train["nb_epoch"]):
        dic_train = get_training_method(trainer, args_train, trainer_ordinary)(
            epoch, 
            loader,
            save_dic = True,
            nb_sample_z_monte_carlo = args_train["nb_sample_z_train_monte_carlo"],
            nb_sample_z_iwae = args_train["nb_sample_z_train_IWAE"],
            loss_function = loss_function,
            verbose = ((epoch+1) % args_train["print_every"] == 0),
        )
        total_dic_train = fill_dic(total_dic_train, dic_train)

        test_this_epoch = args_complete_trainer["save_epoch_function"](epoch, args_train["nb_epoch"])
        if test_this_epoch :
            dic_test = trainer.test(epoch, loader, liste_mc = args_test["liste_mc"])
            if args_output["save_weights"] :
                last_train_loss_in_test = dic_test["train_loss_in_test"]
                if last_train_loss_in_test < best_train_loss_in_test :
                    best_train_loss_in_test = last_train_loss_in_test
                    save_model(final_path, classification_module, selection_module, distribution_module, baseline,)
            total_dic_test = fill_dic(total_dic_test, dic_test)
        
    save_dic(os.path.join(final_path,"train"), total_dic_train)
    save_dic(os.path.join(final_path,"test"), total_dic_test)

    dic_list["train"] = total_dic_train
    dic_list["test"]  = total_dic_test
    

    return final_path, trainer, loader, dic_list 