import os
import torch
import torch.nn as nn

from .classification_training import ordinaryTraining, EVAL_X, trueSelectionTraining
from .interpretation_training import SELECTION_BASED_CLASSIFICATION, REALX
from .selection_training import selectionTraining
from .utils import MSELossLastDim, define_target, continuous_NLLLoss, fill_dic, save_dic, save_model
from .PytorchDistributionUtils.distribution import self_regularized_distributions, RelaxedSubsetSampling, RelaxedBernoulli_thresholded_STE, RelaxedSubsetSampling_STE, L2XDistribution_STE, L2XDistribution
from .Selection.selection_module import SelectionModule, calculate_blocks_patch
from .Prediction import PredictionModule
from .instantiate import *
from .convert_args import *

from torch.distributions import *
from torch.optim import *
import numpy as np
from functools import partial
import pickle as pkl

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def save_parameters(path, complete_args):
    complete_path = os.path.join(path, "parameters")
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    with open(os.path.join(complete_path,"parameters.pkl"), "wb") as f:
        pkl.dump(complete_args, f)





def experiment(dataset, loader, complete_args,):
    if complete_args.args_dataset.train_seed is not None:
        torch.random.manual_seed(complete_args.args_dataset.train_seed )
    dic_list = {}

    ### Prepare output path :
    origin_path = complete_args.args_output.path
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)

    final_path = origin_path

    complete_args_converted = convert_all(complete_args)
    print(vars(complete_args.args_selection))
    print(vars(complete_args_converted.args_selection))
    print("====================================================================================================================================================")
    print(f"Save at {final_path}")
    print(f"Dir at {os.path.dirname(final_path)}")
    print("====================================================================================================================================================")




    ### Regularization method :
    regularization = get_regularization_method(complete_args_converted.args_selection, complete_args_converted.args_distribution_module)
    ## Sampling :
    distribution_module = get_distribution_module_from_args(complete_args_converted.args_distribution_module)
    classification_distribution_module = get_distribution_module_from_args(complete_args_converted.args_classification_distribution_module)
    ### Imputation :
    imputation = get_imputation_method(complete_args_converted.args_classification, dataset)
    ### Networks :
    classifier, selector, baseline, selector_var, reshape_mask_function = get_networks(complete_args_converted.args_classification, complete_args_converted.args_selection, complete_args_converted.args_trainer, dataset.get_dim_output())
    ### Loss Function :
    loss_function = get_loss_function(complete_args_converted.args_train, dataset.get_dim_output())

    
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    save_parameters(final_path,
                    complete_args=complete_args,
                    )
    

    ##### ============ Training POST-HOC ============= ####

    if complete_args_converted.args_train.post_hoc and complete_args_converted.args_train.post_hoc_guidance is not None :
        print("Training post-hoc guidance")
        post_hoc_classifier =  complete_args_converted.args_train.post_hoc_guidance(complete_args_converted.args_selection.input_size_selector, dataset.get_dim_output())
        post_hoc_guidance = PredictionModule(post_hoc_classifier, imputation = imputation)



        trainer = ordinaryTraining(classification_module = post_hoc_guidance,)
        if complete_args_converted.args_train.use_cuda:
            trainer.cuda()

        optim_post_hoc, scheduler_post_hoc = get_optim(post_hoc_guidance, 
                                                    complete_args_converted.args_compiler.optim_post_hoc,
                                                    complete_args_converted.args_compiler.optim_post_hoc_param,
                                                    complete_args_converted.args_compiler.scheduler_post_hoc,
                                                    complete_args_converted.args_compiler.scheduler_post_hoc_param,)
        trainer.compile(optim_classification=optim_post_hoc, scheduler_classification = scheduler_post_hoc)

        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(complete_args_converted.args_train.nb_epoch_post_hoc):
            dic_train = trainer.train_epoch(epoch, loader, loss_function=loss_function, save_dic = True,)
            total_dic_train = fill_dic(total_dic_train, dic_train)

            test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, complete_args_converted.args_train.nb_epoch_post_hoc)
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
    if (complete_args_converted.args_trainer.complete_trainer is EVAL_X) or (complete_args_converted.args_trainer.complete_trainer is REALX) :

            classification_module = PredictionModule(classifier, imputation)

            if  complete_args_converted.args_trainer.complete_trainer is REALX and complete_args_converted.args_train.nb_epoch_pretrain > 0 :
                post_hoc_guidance_eval_x = None
                post_hoc_eval_x = False
            else :
                post_hoc_guidance_eval_x = post_hoc_guidance 
                post_hoc_eval_x = complete_args_converted.args_train.post_hoc


            trainer_ordinary = EVAL_X(classification_module, 
                            reshape_mask_function = reshape_mask_function,
                            post_hoc_guidance = post_hoc_guidance_eval_x,
                            post_hoc = post_hoc_eval_x,
                            argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
                            )
            if complete_args_converted.args_train.use_cuda:
                trainer_ordinary.cuda()

            optim_classification, scheduler_classification = get_optim(classification_module,
                                                            complete_args_converted.args_compiler.optim_classification,
                                                            complete_args_converted.args_compiler.optim_classification_param,
                                                            complete_args_converted.args_compiler.scheduler_classification,
                                                            complete_args_converted.args_compiler.scheduler_classification_param,)
            trainer_ordinary.compile(optim_classification=optim_classification, scheduler_classification = scheduler_classification,)
            
            if complete_args_converted.args_trainer.complete_trainer is EVAL_X :
                nb_epoch = complete_args_converted.args_train.nb_epoch
            else :
                nb_epoch = complete_args_converted.args_train.nb_epoch_pretrain
            total_dic_train = {}
            total_dic_test = {}
            for epoch in range(nb_epoch):
                dic_train = trainer_ordinary.train_epoch(epoch, loader, nb_sample_z_monte_carlo = complete_args_converted.args_train.nb_sample_z_train_monte_carlo, loss_function = loss_function, save_dic = True, verbose=True)
                total_dic_train = fill_dic(total_dic_train, dic_train)
                test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, nb_epoch)
                if test_this_epoch :
                    dic_test = trainer_ordinary.test(epoch, loader, liste_mc = complete_args_converted.args_test.liste_mc)
                    total_dic_test = fill_dic(total_dic_test, dic_test)


            dic_list["train_pretraining_eval_x"] = total_dic_train
            dic_list["test_pretraining_eval_x"]  = total_dic_test

            if complete_args_converted.args_trainer.complete_trainer is EVAL_X:
                dic_list["train"] = total_dic_train
                dic_list["test"]  = total_dic_test
                save_dic(os.path.join(final_path,"train"), total_dic_train)
                save_dic(os.path.join(final_path,"test"), total_dic_test)
                return final_path, trainer_ordinary, loader, dic_list

    else :
        if complete_args_converted.args_trainer.complete_trainer is ordinaryTraining or complete_args_converted.args_trainer.complete_trainer is trueSelectionTraining :
            nb_epoch = int(complete_args_converted.args_train.nb_epoch)
            post_hoc_guidance_ordinary = post_hoc_guidance
            post_hoc_ordinary = complete_args_converted.args_train.post_hoc
            
        else :
            nb_epoch = complete_args_converted.args_train.nb_epoch_pretrain
            post_hoc_guidance_ordinary = None
            post_hoc_ordinary = False


        vanilla_classification_module = PredictionModule(classifier,  imputation = imputation)

        if complete_args_converted.args_trainer.complete_trainer is trueSelectionTraining :
            trainer_ordinary = trueSelectionTraining(vanilla_classification_module, 
                                    post_hoc_guidance = post_hoc_guidance_ordinary,
                                    post_hoc = post_hoc_ordinary,
                                    argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
                                    )
            true_selection = True
        else :
            trainer_ordinary = ordinaryTraining(vanilla_classification_module,
                                    post_hoc_guidance = post_hoc_guidance_ordinary,
                                    post_hoc = post_hoc_ordinary,
                                    argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
                                )
            true_selection = False

        if complete_args_converted.args_train.use_cuda:
            trainer_ordinary.cuda()

        optim_classification, scheduler_classification = get_optim(vanilla_classification_module,
                                                                complete_args_converted.args_compiler.optim_classification,
                                                                complete_args_converted.args_compiler.optim_classification_param,
                                                                complete_args_converted.args_compiler.scheduler_classification,
                                                                complete_args_converted.args_compiler.scheduler_classification_param,)
        trainer_ordinary.compile(optim_classification=optim_classification, scheduler_classification = scheduler_classification,)


        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(int(nb_epoch)):
            dic_train = trainer_ordinary.train_epoch(epoch, loader, loss_function = loss_function, save_dic = True, verbose=True)
            total_dic_train = fill_dic(total_dic_train, dic_train)
            test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, nb_epoch)
            if test_this_epoch :
                if true_selection :
                    dic_test = trainer_ordinary.test(epoch, loader, liste_mc = complete_args_converted.args_test.liste_mc)
                else :
                    dic_test = trainer_ordinary.test(epoch, loader,)
                total_dic_test = fill_dic(total_dic_test, dic_test)


        if complete_args_converted.args_trainer.complete_trainer is ordinaryTraining or complete_args_converted.args_trainer.complete_trainer is trueSelectionTraining :
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
                    activation=complete_args_converted.args_selection.activation,
                    regularization=regularization,
                    )

    if complete_args_converted.args_train.nb_epoch_pretrain_selector > 0 :
        selection_trainer = selectionTraining(selection_module, complete_args_converted.args_train.use_regularization_pretrain_selector)
        optim_selection, scheduler_selection = get_optim(selection_module,
                                            complete_args_converted.args_compiler.optim_selection,
                                            complete_args_converted.args_compiler.optim_selection_param,
                                            complete_args_converted.args_compiler.scheduler_selection,
                                            complete_args_converted.args_compiler.scheduler_selection_param,)
        selection_trainer.compile(optim_selection=optim_selection, scheduler_selection = scheduler_selection,)
        nb_epoch = complete_args_converted.args_train.nb_epoch_pretrain_selector
    
        if complete_args_converted.args_train.use_cuda:
            selection_trainer.cuda()

        if loader.dataset.optimal_S_train is None or loader.dataset.optimal_S_test is None :
            raise AttributeError("optimal_S_train or optimal_S_test not define for this dataset")

        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(int(nb_epoch)):
            dic_train = selection_trainer.train_epoch(epoch, loader, save_dic = True,)
            total_dic_train = fill_dic(total_dic_train, dic_train)

            test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, nb_epoch)
            if test_this_epoch :
                dic_test = selection_trainer.test(epoch, loader, )
                total_dic_test = fill_dic(total_dic_test, dic_test)

        
            
        dic_list["train_selection_pretraining"] = total_dic_train
        dic_list["test_selection_pretaining"]  = total_dic_test


                


    ##### ============  Modules initialisation for complete training ===========:
   


    classification_module = PredictionModule(classifier, imputation=imputation)
    trainer = complete_args_converted.args_trainer.complete_trainer(
        classification_module,
        selection_module,
        monte_carlo_gradient_estimator = complete_args_converted.args_trainer.monte_carlo_gradient_estimator,
        baseline = baseline,
        distribution_module = distribution_module,
        classification_distribution_module = classification_distribution_module,
        reshape_mask_function = reshape_mask_function,
        fix_classifier_parameters = complete_args_converted.args_train.fix_classifier_parameters,
        fix_selector_parameters = complete_args_converted.args_train.fix_selector_parameters,
        post_hoc_guidance = post_hoc_guidance,
        post_hoc = complete_args_converted.args_train.post_hoc,
        argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
        )

    if complete_args_converted.args_train.use_cuda:
        trainer.cuda()

    ####Optim_optim_classification :

    optim_classification, scheduler_classification = get_optim(classification_module,
                                                    complete_args_converted.args_compiler.optim_classification,
                                                    complete_args_converted.args_compiler.optim_classification_param,
                                                    complete_args_converted.args_compiler.scheduler_classification,
                                                    complete_args_converted.args_compiler.scheduler_classification_param,) 
    optim_selection, scheduler_selection = get_optim(selection_module,
                                            complete_args_converted.args_compiler.optim_selection,
                                            complete_args_converted.args_compiler.optim_selection_param,
                                            complete_args_converted.args_compiler.scheduler_selection,
                                            complete_args_converted.args_compiler.scheduler_selection_param,)
    optim_baseline, scheduler_baseline = get_optim(baseline,
                                        complete_args_converted.args_compiler.optim_baseline,
                                        complete_args_converted.args_compiler.optim_baseline_param,
                                        complete_args_converted.args_compiler.scheduler_baseline,
                                        complete_args_converted.args_compiler.scheduler_baseline_param,)
    optim_distribution_module, scheduler_distribution_module = get_optim(distribution_module,
                                                                complete_args_converted.args_compiler.optim_distribution_module,
                                                                complete_args_converted.args_compiler.optim_distribution_module_param,
                                                                complete_args_converted.args_compiler.scheduler_distribution_module,
                                                                complete_args_converted.args_compiler.scheduler_distribution_module_param,)

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
    for epoch in range(complete_args_converted.args_train.nb_epoch):
        dic_train = get_training_method(trainer, complete_args_converted.args_train, trainer_ordinary)(
            epoch, 
            loader,
            save_dic = True,
            nb_sample_z_monte_carlo = complete_args_converted.args_train.nb_sample_z_train_monte_carlo,
            nb_sample_z_iwae = complete_args_converted.args_train.nb_sample_z_train_IWAE,
            loss_function = loss_function,
            verbose = ((epoch+1) % complete_args_converted.args_train.print_every == 0),
        )
        total_dic_train = fill_dic(total_dic_train, dic_train)

        test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, complete_args_converted.args_train.nb_epoch)
        if test_this_epoch :
            dic_test = trainer.test(epoch, loader, args = complete_args, liste_mc = complete_args_converted.args_test.liste_mc,)
            if complete_args_converted.args_output.save_weights :
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