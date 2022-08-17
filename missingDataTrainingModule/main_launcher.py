import os
import torch
import torch.nn as nn

from .utils import fill_dic, save_dic, save_model
from .Selection import SelectionModule
from .Prediction import PredictionModule
from .Trainer import ordinaryPredictionTraining, trainingWithSelection, selectionTraining
from .EpochsScheduler import classic_train_epoch, alternate_ordinary_train_epoch, alternate_fixing_train_epoch
from .instantiate import *
from .convert_args import *
from .EvaluationUtils import test_epoch

from torch.distributions import *
from torch.optim import *
import numpy as np
from functools import partial
import pickle as pkl
from utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def save_parameters(path, complete_args):
    complete_path = os.path.join(path, "parameters")
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    with open(os.path.join(complete_path,"parameters.pkl"), "wb") as f:
        pkl.dump(complete_args, f)

    with open(os.path.join(complete_path,"parameters.txt"), "w") as f:
        dic = from_args_to_dictionary(complete_args, to_str = False)
        f.write(dic_to_line_str(dic,))




def experiment(dataset, loader, complete_args,):
    if complete_args.args_dataset.args_dataset_parameters.train_seed is not None:
        torch.random.manual_seed(complete_args.args_dataset.args_dataset_parameters.train_seed )
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
    classifier, selector, baseline, selector_var, reshape_mask_function = get_networks(complete_args_converted.args_classification,
                                                                        complete_args_converted.args_selection,
                                                                        complete_args_converted.args_trainer,
                                                                        complete_args_converted.args_interpretable_module,
                                                                        dataset=dataset)
    ### Loss Function :
    loss_function = get_loss_function(complete_args_converted.args_train.loss_function,
                                            complete_args_converted.args_train,
                                            dataset.get_dim_output())
    loss_function_selection = get_loss_function(complete_args_converted.args_train.loss_function_selection,
                                            complete_args_converted.args_train,
                                            dataset.get_dim_output())
    ### Complete Module :
    prediction_module = PredictionModule(classifier, imputation=imputation)
    selection_module =  SelectionModule(selector,
                    activation=complete_args_converted.args_selection.activation,
                    regularization=regularization,
                    )
    selection_module_var = None
    interpretable_module = get_complete_module(complete_args_converted.args_interpretable_module.interpretable_module,
                                                prediction_module,
                                                selection_module,
                                                selection_module_var,
                                                distribution_module = distribution_module,
                                                classification_distribution_module = classification_distribution_module,
                                                reshape_mask_function = reshape_mask_function,
                                                dataset = dataset,
                                                )
    


    if not os.path.exists(final_path):
        os.makedirs(final_path)

    save_parameters(final_path,
                    complete_args=complete_args,
                    )
    

    ##### ============ Training POST-HOC ============= ####

    if complete_args_converted.args_train.post_hoc and complete_args_converted.args_train.post_hoc_guidance is not None :
        print("Training post-hoc guidance")
        post_hoc_classifier =  complete_args_converted.args_train.post_hoc_guidance(complete_args_converted.args_selection.input_size_selector, dataset.get_dim_output())
        post_hoc_guidance_prediction_module = PredictionModule(post_hoc_classifier, imputation = imputation)
        post_hoc_guidance_complete_module = PredictionCompleteModel(post_hoc_guidance_prediction_module,)

        trainer = ordinaryPredictionTraining(prediction_module = post_hoc_guidance_complete_module, loss_function = loss_function,)
        if complete_args_converted.args_train.use_cuda:
            trainer.cuda()

        optim_post_hoc, scheduler_post_hoc = get_optim(post_hoc_guidance_prediction_module, 
                                                    complete_args_converted.args_compiler.optim_post_hoc,
                                                    complete_args_converted.args_compiler.optim_post_hoc_param,
                                                    complete_args_converted.args_compiler.scheduler_post_hoc,
                                                    complete_args_converted.args_compiler.scheduler_post_hoc_param,)
        trainer.compile(optim_classification=optim_post_hoc, scheduler_classification = scheduler_post_hoc)

        epoch_scheduler = classic_train_epoch(save_dic = True, verbose=True,)
        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(complete_args_converted.args_train.nb_epoch_post_hoc):
            dic_train = epoch_scheduler(epoch, loader, trainer)
            total_dic_train = fill_dic(total_dic_train, dic_train)

            test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, complete_args_converted.args_train.nb_epoch_post_hoc)
            if test_this_epoch :
                dic_test = test_epoch(interpretable_module, epoch, loader, args = complete_args, liste_mc = [], trainer = trainer)
                total_dic_test = fill_dic(total_dic_test, dic_test)

        save_dic(os.path.join(final_path,"train_post_hoc"), total_dic_train)
        save_dic(os.path.join(final_path,"test_post_hoc"), total_dic_test)

        dic_list["train_post_hoc"] = total_dic_train
        dic_list["test_post_hoc"]  = total_dic_test
        post_hoc_guidance_prediction_module.eval()
    else :
        post_hoc_guidance_prediction_module = None

    ##### ============ Modules initialisation for ordinary training ============:

    pretrainer_pred = None
    if complete_args_converted.args_train.nb_epoch_pretrain > 0 :
        if (complete_args_converted.args_interpretable_module.interpretable_module is DECOUPLED_SELECTION or complete_args_converted.args_interpretable_module.interpretable_module is COUPLED_SELECTION):
            if complete_args_converted.args_interpretable_module.interpretable_module is DECOUPLED_SELECTION :
                pretrainer_pred = trainingWithSelection(interpretable_module.EVALX, 
                                post_hoc_guidance = post_hoc_guidance_prediction_module,
                                post_hoc = complete_args_converted.args_train.post_hoc,
                                argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
                                loss_function = loss_function,
                                nb_sample_z_monte_carlo = complete_args_converted.args_train.nb_sample_z_train_monte_carlo_classification,
                                nb_sample_z_iwae = complete_args_converted.args_train.nb_sample_z_train_IWAE_classification,
                                )
            elif complete_args_converted.args_interpretable_module.interpretable_module is COUPLED_SELECTION :
                pretrainer_pred = ordinaryPredictionTraining(interpretable_module,
                                post_hoc_guidance = post_hoc_guidance_prediction_module,
                                post_hoc = complete_args_converted.args_train.post_hoc,
                                argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
                                loss_function = loss_function,
                                )
            
            if complete_args_converted.args_train.use_cuda:
                pretrainer_pred.cuda()

            optim_classification, scheduler_classification = get_optim(interpretable_module.prediction_module,
                                                            complete_args_converted.args_compiler.optim_classification,
                                                            complete_args_converted.args_compiler.optim_classification_param,
                                                            complete_args_converted.args_compiler.scheduler_classification,
                                                            complete_args_converted.args_compiler.scheduler_classification_param,)
            pretrainer_pred.compile(optim_classification=optim_classification, scheduler_classification = scheduler_classification,)
            nb_epoch = complete_args_converted.args_train.nb_epoch_pretrain

            epoch_scheduler = classic_train_epoch(save_dic = True, verbose=complete_args_converted.args_train.verbose,)
            total_dic_train = {}
            total_dic_test = {}
            for epoch in range(nb_epoch):
                dic_train = epoch_scheduler(epoch, loader, pretrainer_pred)
                total_dic_train = fill_dic(total_dic_train, dic_train)

                test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, complete_args_converted.args_train.nb_epoch_post_hoc)
                if test_this_epoch :
                    dic_test = test_epoch(interpretable_module, epoch, loader, args = complete_args, liste_mc = [], trainer = pretrainer_pred)
                    total_dic_test = fill_dic(total_dic_test, dic_test)

            dic_list["train_pretraining"] = total_dic_train
            dic_list["test_pretraining"]  = total_dic_test
        else :
            print(f"Pretraining is either not implemented or not needed for this interpretable module {complete_args.args_interpretable_module.interpretable_module}")
            

    ##### ======================= Training in selection ==========================:


    if complete_args_converted.args_train.nb_epoch_pretrain_selector > 0 :
        selection_trainer = selectionTraining(interpretable_module, complete_args_converted.args_train.use_regularization_pretrain_selector)
        optim_selection, scheduler_selection = get_optim(interpretable_module.selection_module,
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
   

    trainer = get_trainer(complete_args_converted.args_trainer.complete_trainer,
        interpretable_module,
        monte_carlo_gradient_estimator = complete_args_converted.args_trainer.monte_carlo_gradient_estimator,
        baseline = None,
        fix_classifier_parameters = complete_args_converted.args_train.fix_classifier_parameters,
        fix_selector_parameters = complete_args_converted.args_train.fix_selector_parameters,
        post_hoc = complete_args_converted.args_train.post_hoc,
        post_hoc_guidance = post_hoc_guidance_prediction_module,
        argmax_post_hoc = complete_args_converted.args_train.argmax_post_hoc,
        loss_function = loss_function,
        loss_function_selection = loss_function_selection,
        nb_sample_z_monte_carlo = complete_args_converted.args_train.nb_sample_z_train_monte_carlo,
        nb_sample_z_iwae = complete_args_converted.args_train.nb_sample_z_train_IWAE,
        nb_sample_z_monte_carlo_classification = complete_args_converted.args_train.nb_sample_z_train_monte_carlo_classification,
        nb_sample_z_iwae_classification = complete_args_converted.args_train.nb_sample_z_train_IWAE_classification,
        )



    if complete_args_converted.args_train.use_cuda:
        trainer.cuda()

    ####Optim_optim_classification :

    optim_classification, scheduler_classification = get_optim(prediction_module,
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

    trainer = compile_trainer(trainer,
        trainer_type = complete_args_converted.args_trainer.complete_trainer,
        optim_classification = optim_classification,
        optim_selection = optim_selection,
        scheduler_classification = scheduler_classification,
        scheduler_selection = scheduler_selection,
        optim_baseline = optim_baseline,
        scheduler_baseline = scheduler_baseline,
        optim_distribution_module = optim_distribution_module,
        scheduler_distribution_module = scheduler_distribution_module,
        )



###### Main module training :
    # epoch_scheduler = classic_train_epoch(save_dic = True, verbose=((epoch+1) % complete_args_converted.args_train.print_every == 0),)
    epoch_scheduler = classic_train_epoch(save_dic = True, verbose=complete_args_converted.args_train.verbose,)
    best_train_loss_in_test = float("inf")
    total_dic_train = {}
    total_dic_test = {}
    for epoch in range(complete_args_converted.args_train.nb_epoch):
        dic_train = epoch_scheduler(epoch, loader, trainer)
        total_dic_train = fill_dic(total_dic_train, dic_train)

        test_this_epoch = complete_args_converted.args_trainer.save_epoch_function(epoch, complete_args_converted.args_train.nb_epoch)
        if test_this_epoch :
            dic_test = test_epoch(interpretable_module, epoch, loader, args = complete_args, liste_mc = complete_args_converted.args_test.liste_mc, trainer = trainer)
            total_dic_test = fill_dic(total_dic_test, dic_test)
            if complete_args_converted.args_output.save_weights :
                last_train_loss_in_test = dic_test["train_loss_in_test"]
                if last_train_loss_in_test < best_train_loss_in_test :
                    best_train_loss_in_test = last_train_loss_in_test
                    save_model(final_path, prediction_module, selection_module, distribution_module, baseline,suffix = "_best")
        
        
    save_model(final_path, prediction_module, selection_module, distribution_module, baseline,suffix = "_last")

    save_dic(os.path.join(final_path,"train"), total_dic_train)
    save_dic(os.path.join(final_path,"test"), total_dic_test)

    dic_list["train"] = total_dic_train
    dic_list["test"]  = total_dic_test
    

    return final_path, trainer, loader, dic_list 