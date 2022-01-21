import numpy as np
from psutil import net_connections


from .classicTraining import *
from .utils import *


from torch.distributions import *
from torch.optim import *
from functools import partial

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def save_parameters(path, args_classification, args_selection, args_complete_trainer, args_train, args_test, args_output, args_compiler):
    complete_path = os.path.join(path, "parameters")
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    with open(os.path.join(complete_path,"classification.txt"), "w") as f:
        f.write(str(args_classification))
    
    with open(os.path.join(complete_path,"selection.txt"), "w") as f:
        f.write(str(args_selection))

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

def comply_size(dataset, args_classification, args_selection, args_complete_trainer):
    dim_shape = dataset.get_dim_input()
    args_classification["input_size_classification_module"] = dim_shape # Size before imputation
    args_classification["input_size_classifier"] = dim_shape # Size after imputation
    args_classification["input_size_baseline"] = dim_shape # Size before imputation (should be size of data)
    args_selection["input_size_selector"] = dim_shape
    args_selection["output_size_selector"] = dim_shape
    args_selection["input_size_autoencoder"] = dim_shape
    args_complete_trainer["input_size_baseline"] = dim_shape
    args_complete_trainer["reshape_mask_function"] = partial(args_complete_trainer["reshape_mask_function"], size = dim_shape)

def get_imputation_method(args_classification, dataset):
    if args_classification["mask_reg"] is not None :
        mask_reg = args_classification["mask_reg"](rate = args_classification["mask_reg_rate"])
    else :
        mask_reg = None
    
    if args_classification["post_process_regularization"] is not None :
        post_process_regularization = args_classification["post_process_regularization"](
                                        network_post_process = args_classification["network_post_process"],
                                        trainable = args_classification["post_process_trainable"],
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
                                        nb_imputation = args_classification["nb_imputation"],
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




def get_networks(args_classification, args_selection, args_complete_trainer, output_category):
    input_size_selector = args_selection["input_size_selector"]
    output_size_selector = args_selection["output_size_selector"]
    input_size_classifier = args_classification["input_size_classification_module"]
    input_size_baseline = args_classification["input_size_classification_module"]

    classifier =  args_classification["classifier"](input_size_classifier, output_category)
    selector = args_selection["selector"](input_size_selector, output_size_selector)
    

    if args_selection["selector_var"] is not None :
        selector_var = args_selection["selector_var"](input_size_selector, output_size_selector)
    else :
        selector_var = None

    
    if args_complete_trainer["baseline"] is not None :
        baseline = args_complete_trainer["baseline"](input_size_baseline, output_category)
    else :
        baseline = None
 
    return classifier, selector, baseline, selector_var

def get_regularization_method(args_selection):
    regularization = args_selection["regularization"](**args_selection)
    return regularization



def check_parameters_compatibility(args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_output):
    sampling_distrib = args_distribution_module["distribution"]
    activation = args_selection["activation"]
    if args_distribution_module["distribution"] in [RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2X_Distribution_STE, L2X_Distribution] \
        and args_selection["activation"] != torch.nn.LogSoftmax() :
        raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")
    
    # if args_distribution_module["distribution"] in [RelaxedBernoulli_thresholded_STE, RelaxedBernoulli] \
        # and args_selection["activation"] != torch.nn.LogSigmoid() :
        # raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")


def experiment(dataset, loader, args_output, args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = False):
    torch.random.manual_seed(0)
    dic_list = {}

    ### Prepare output path :
    origin_path = args_output["path"]
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)

    if name_modification :
        print("Modification of the name")
        folder = os.path.join(origin_path,str(dataset))
        if not os.path.exists(folder):
            os.makedirs(folder)

        experiment_name = args_output["experiment_name"]
        final_path = os.path.join(folder, experiment_name)
    else :
        final_path = origin_path


    if not os.path.exists(final_path):
        os.makedirs(final_path)

    
    print(f"Save at {final_path}")
    save_parameters(final_path, args_classification = args_classification,
                    args_selection = args_selection,
                    args_complete_trainer= args_complete_trainer,
                    args_train = args_train,
                    args_test = args_test,
                    args_output=args_output,
                    args_compiler=args_compiler)
    check_parameters_compatibility(args_classification, args_selection, args_distribution_module, args_complete_trainer, args_train, args_test, args_output)

    ### Datasets :

    if args_complete_trainer["comply_with_dataset"] :
        comply_size(dataset, args_classification, args_selection, args_complete_trainer)

    ## Sampling :
    distribution_module = get_distribution_module_from_args(args_distribution_module)

    ### Imputation :
    imputation = get_imputation_method(args_classification, dataset)

    ### Regularization method :
    regularization = get_regularization_method(args_selection)

    ### Networks :
    classifier, selector, baseline, selector_var = get_networks(args_classification, args_selection, args_complete_trainer, dataset.get_dim_output())

    post_hoc_guidance = None


    ##### ============ Training POST-HOC ============= ####

    if args_train["post_hoc"] and args_train["post_hoc_guidance"] is not None :
        print("Training post-hoc guidance")
        post_hoc_classifier =  args_train["post_hoc_guidance"](args_selection["input_size_selector"], dataset.get_dim_output())
        post_hoc_guidance = ClassificationModule(post_hoc_classifier, imputation = imputation)
        optim_post_hoc = args_train["optim_post_hoc"](post_hoc_guidance.parameters())

        if args_train["scheduler_post_hoc"] is not None :
            scheduler_post_hoc = args_train["scheduler_post_hoc"](optim_post_hoc)
        else :
            scheduler_post_hoc = None
        trainer = ordinaryTraining(classification_module = post_hoc_guidance,)
        if args_train["use_cuda"]:
            trainer.cuda()
        trainer.compile(optim_classification=optim_post_hoc, scheduler_classification = scheduler_post_hoc)

        nb_epoch = args_train["nb_epoch_post_hoc"]
        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(nb_epoch):
            dic_train = trainer.train_epoch(epoch, loader, save_dic = True, print_dic_bool= ((epoch+1) % args_train["print_every"] == 0),)
            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch == nb_epoch-1:
                dic_test = trainer.test(loader)

            
            total_dic_train = fill_dic(total_dic_train, dic_train)
            total_dic_test = fill_dic(total_dic_test, dic_test)
            
        save_dic(os.path.join(final_path,"train_post_hoc"), total_dic_train)
        save_dic(os.path.join(final_path,"test_post_hoc"), total_dic_test)

        dic_list["train_post_hoc"] = total_dic_train
        dic_list["test_post_hoc"]  = total_dic_test


    ##### ============ Modules initialisation for ordinary training ============:
    # if args_complete_trainer["complete_trainer"] is REALX and args_train["nb_epoch_pretrain"] > 0 :
        
    #     # if os.path.exists(os.path.join(args_output["path"], "weightsClassifier.pt")):
    #     #     print(classifier)

    #     #     classifier.load_state_dict(torch.load(os.path.join(args_output["path"], "weightsClassifier.pt")))
    #     #     print("LOADED")
    #     # else :
    #         imputation = imputation(input_size= args_classification["input_size_classification_module"], post_process_regularization = post_proc_regul,
    #                         reconstruction_reg= None, )


    #         classification_module = ClassificationModule(classifier, imputation)
    #         optim_classification = args_compiler["optim_classification"](classification_module.parameters())

    #         if args_compiler["scheduler_classification"] is not None :
    #             scheduler_classification = args_compiler["scheduler_classification"](optim_classification)
    #         else :
    #             scheduler_classification = None

    #         trainer = EVAL_X(classification_module, fixed_distribution= FixedBernoulli(), reshape_mask_function = args_complete_trainer["reshape_mask_function"])
    #         if args_train["use_cuda"]:
    #             trainer.cuda()
    #         trainer.compile(optim_classification=optim_classification, scheduler_classification = scheduler_classification,)
            
    #         for epoch in range(args_train["nb_epoch_pretrain"]):
    #             dic_train = trainer.train_epoch(epoch, loader, save_dic = True, nb_sample_z_monte_carlo = args_train["nb_sample_z_train_monte_carlo"],)
    #             if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch == nb_epoch-1:
    #                 dic_test = trainer.test(loader,)

    #         with open( os.path.join(args_output["path"], "weightsClassifier.pt"), 'wb') as f:
    #             torch.save(classifier.state_dict(), f)
                    


    elif args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trainingWithSelection  or args_complete_trainer["complete_trainer"] is trainingWithSelection or args_train["nb_epoch_pretrain"]>0 :
        
        if args_complete_trainer["complete_trainer"] is trainingWithSelection :
            assert Imputation is SelectionAsInput
            
        if args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trainingWithSelection :
            nb_epoch = int(args_train["nb_epoch"])
        else :
            nb_epoch = args_train["nb_epoch_pretrain"]

        vanilla_classification_module = ClassificationModule(classifier, imputation = imputation)
        if args_train["use_cuda"]:
            vanilla_classification_module.cuda()
        optim_classifier = args_compiler["optim_classification"](vanilla_classification_module.parameters())

        if args_compiler["scheduler_classification"] is not None :
            scheduler_classification = args_compiler["scheduler_classification"](optim_classifier)
        else :
            scheduler_classification = None

        trainer = ordinaryTraining(vanilla_classification_module, )
        if args_train["use_cuda"]:
            trainer.cuda()
        trainer.compile(optim_classification=optim_classifier, scheduler_classification = scheduler_classification,)


        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(int(nb_epoch)):
            dic_train = trainer.train_epoch(epoch, loader, save_dic = True,)
            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch == nb_epoch-1:
                dic_test = trainer.test(loader) 
            total_dic_train = fill_dic(total_dic_train, dic_train)
            total_dic_test = fill_dic(total_dic_test, dic_test)
            
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test"), total_dic_test)

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test

        if args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trainingWithSelection:
            return final_path, trainer, loader, dic_list
            
    
    #### Pretraining selector :
    if args_train["nb_epoch_pretrain_selector"] >0:
        print("Pretraining selector")
        selection_module = SelectionModule(selector,
                            activation=args_selection["activation"],
                            regularization=regularization,
                            )

        optim_selection = args_compiler["optim_selection"](selection_module.parameters())
        if args_compiler["scheduler_selection"] is not None :
            scheduler_selection = args_compiler["scheduler_selection"](optim_selection)
        else :
            scheduler_selection = None
        trainer_selector = GroundTruthSelectionTraining(selection_module, reshape_mask_function=args_complete_trainer["reshape_mask_function"])
        if args_train["use_cuda"]:
            trainer_selector.cuda()
        trainer_selector.compile(optim_selection=optim_selection, scheduler_selection = scheduler_selection)
        for epoch in range(args_train["nb_epoch_pretrain_selector"]):
            trainer_selector.train_epoch(epoch, loader, verbose=True)
            trainer_selector.test(epoch, loader)
                


    ##### ============  Modules initialisation for complete training ===========:
    if args_complete_trainer["complete_trainer"] is not ordinaryTraining and args_complete_trainer["complete_trainer"] is not trainingWithSelection :

        if args_classification["reconstruction_regularization"] is not None :
            recons_regul = args_classification["reconstruction_regularization"](args_classification["autoencoder"], to_train = args_classification["train_reconstruction_regularization"])
        else : 
            recons_regul = None

        selection_module = SelectionModule(selector,
                            activation=args_selection["activation"],
                            regularization=regularization,
                            )

        classification_module = ClassificationModule(classifier, imputation=imputation)
        




        trainer = args_complete_trainer["complete_trainer"](
            classification_module,
            selection_module,
            distribution_module = distribution_module,
            baseline = baseline,
            reshape_mask_function = args_complete_trainer["reshape_mask_function"],
            fix_classifier_parameters = args_train["fix_classifier_parameters"],
            post_hoc_guidance = post_hoc_guidance,
            post_hoc = args_train["post_hoc"],
            argmax_post_hoc = args_train["argmax_post_hoc"],
        )

        if args_train["use_cuda"]:
            trainer.cuda()

        ####Optim_optim_classification :
        optim_classification = args_compiler["optim_classification"](classification_module.parameters(), weight_decay = 1e-5)
        if args_compiler["scheduler_classification"] is not None :
            scheduler_classification = args_compiler["scheduler_classification"](optim_classification)
        else :
            scheduler_classification = None


        optim_selection = args_compiler["optim_selection"](selection_module.parameters(), weight_decay = 1e-5)
        if args_compiler["scheduler_selection"] is not None :
            scheduler_selection = args_compiler["scheduler_selection"](optim_selection)
        else :
            scheduler_selection = None
        
        if args_complete_trainer["baseline"] is not None :
            optim_baseline = args_compiler["optim_baseline"](baseline.parameters(), weight_decay = 1e-5)
            scheduler_baseline = args_compiler["scheduler_baseline"](optim_baseline)
        else :
            optim_baseline = None
            scheduler_baseline = None

        if args_compiler["optim_distribution_module"] is not None and len(list(distribution_module.parameters())) > 0 :
                optim_distribution_module = args_compiler["optim_distribution_module"](distribution_module.parameters(), weight_decay = 1e-5)
                scheduler_distribution_module = args_compiler["scheduler_distribution_module"](optim_distribution_module)
        else :
            optim_distribution_module = None
            scheduler_distribution_module = None

        trainer.compile(optim_classification = optim_classification,
            optim_selection = optim_selection,
            scheduler_classification = scheduler_classification,
            scheduler_selection = scheduler_selection,
            optim_baseline = optim_baseline,
            scheduler_baseline = scheduler_baseline,
            optim_distribution_module = optim_distribution_module,
            scheduler_distribution_module = scheduler_distribution_module,)



######============== Complete module training================:


        total_dic_train = {}
        total_dic_test = {}
        total_dic_test_no_var_2 = {}
        total_dic_test_var = {}
        for epoch in range(args_train["nb_epoch"]):
            dic_train = trainer.train_epoch(
                epoch, loader,
                save_dic = True,
                nb_sample_z_monte_carlo =args_train["nb_sample_z_train_monte_carlo"],
                nb_sample_z_IWAE = args_train["nb_sample_z_train_IWAE"],
                verbose = ((epoch+1) % args_train["print_every"] == 0),
            )

            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch ==args_train["nb_epoch"]-1:
                dic_test = trainer.test(loader, nb_sample_z=args_test["nb_sample_z_test"])
                total_dic_train = fill_dic(total_dic_train, dic_train)
                total_dic_test = fill_dic(total_dic_test, dic_test)
           
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test"), total_dic_test)

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test
        

    return final_path, trainer, loader, dic_list 