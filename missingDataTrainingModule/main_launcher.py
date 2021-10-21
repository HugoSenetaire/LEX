# from .Destruction import * 
# from .Classification import *
# from .utils_missing import *
import numpy as np
from .completeTrainer import *
from .utils import *


from torch.distributions import *
from torch.optim import *
from functools import partial

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def save_parameters(path, args_dataset, args_classification, args_destruction, args_complete_trainer, args_train, args_test, args_output):
    complete_path = os.path.join(path, "parameters")
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    with open(os.path.join(complete_path,"dataset.txt"), "w") as f:
        f.write(str(args_dataset))

    with open(os.path.join(complete_path,"classification.txt"), "w") as f:
        f.write(str(args_classification))
    
    with open(os.path.join(complete_path,"destruction.txt"), "w") as f:
        f.write(str(args_destruction))

    with open(os.path.join(complete_path,"complete_trainer.txt"), "w") as f:
        f.write(str(args_complete_trainer))

    with open(os.path.join(complete_path,"train.txt"), "w") as f:
        f.write(str(args_train))

    with open(os.path.join(complete_path,"test.txt"), "w") as f:
        f.write(str(args_test))

    with open(os.path.join(complete_path,"output.txt"), "w") as f:
        f.write(str(args_output))

def get_dataset(args_dataset, args_train, args_classification, args_destruction):
    if "HypercubeDataset" in str(args_dataset["dataset"]):
        dataset = partial(args_dataset["dataset"],
                         nb_shape = args_dataset["nb_shape"],
                         nb_dim = args_dataset["nb_dim"],
                         ratio_sigma = args_dataset["ratio_sigma"],
                         sigma = args_dataset["sigma"],
                         prob_simplify = args_dataset["prob_simplify"],
                         nb_sample_train = args_dataset["nb_samples_train"],
                         nb_sample_test = args_dataset["nb_samples_test"],
                         give_index = args_dataset["give_index"],
                         centroids_path = args_dataset["centroids_path"],
                         generate_new = args_dataset["generate_new"],
                         use_cuda= args_train["use_cuda"],
                         save = args_dataset["save"],
                         generate_each_time = args_dataset["generate_each_time"],
                         )
        loader = args_dataset["loader"](dataset, root_dir = args_dataset["root_dir"], batch_size_test=args_dataset["batch_size_test"], batch_size_train=args_dataset["batch_size_train"], nb_sample_train = args_dataset["nb_samples_train"], nb_sample_test = args_dataset["nb_samples_test"])
        nb_dim = loader.dataset.nb_dim
        args_classification["input_size_classification_module"] = (1,nb_dim) # Size before imputation
        args_classification["input_size_classifier"] = (1,nb_dim) # Size after imputation
        args_classification["input_size_classifier_baseline"] = (1,nb_dim) # Size before imputation (should be size of data)
        args_destruction["input_size_destructor"] = (1,nb_dim)
        args_destruction["input_size_autoencoder"] = (1,nb_dim)
    else :
        dataset = partial(args_dataset["dataset"], give_index = args_dataset["give_index"])
        loader = args_dataset["loader"](dataset, root_dir = args_dataset["root_dir"], batch_size_test=args_dataset["batch_size_test"], batch_size_train=args_dataset["batch_size_train"])
    return dataset, loader


def get_imputation_method(args_class):
    
    if args_class["imputation"] is ConstantImputation or args_class["imputation"] is ConstantImputation_ContinuousAddMask or args_class["imputation"] is ConstantImputation_ContinuousAddMaskV2:
        return partial(args_class["imputation"], cste = args_class["cste_imputation"], add_mask = args_class["add_mask"])
    elif args_class["imputation"] is LearnConstantImputation:
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])
    elif args_class["imputation"] is NoiseImputation :
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])
    else :
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])


def get_multiple_imputation(args_classification, args_train, loader):
    post_process_regularization = args_classification["post_process_regularization"]
    if args_classification["post_process_regularization"] is VAEAC_Imputation_DetachVersion :
        model, sampler = load_VAEAC(args_classification["VAEAC_dir"])
        post_proc_regul = post_process_regularization(model, sampler, args_classification["nb_imputation"])
    elif post_process_regularization is DatasetBasedImputation :
        post_proc_regul = post_process_regularization(loader.dataset, args_classification["nb_imputation"])
    elif args_classification["post_process_regularization"] is MICE_imputation :
        post_proc_regul = post_process_regularization(args_classification["nb_imputation"])
    elif args_classification["post_process_regularization"] is MarkovChainImputation :
        print("Training Markov Chain")
        markov_chain = MarkovChain(loader.train_loader, use_cuda=args_train["use_cuda"])
        post_proc_regul = post_process_regularization(markov_chain, args_classification["nb_imputation"], use_cuda=args_train["use_cuda"])
    elif args_classification["post_process_regularization"] is HMMimputation:
        print("Train HMM")
        if args_classification["log_hmm"] :
            hmm = HMMLog(loader.train_loader, hidden_dim = args_classification["hidden_state_hmm"],
                    nb_iter = args_classification["nb_iter_hmm"], nb_start = args_classification["nb_start_hmm"], use_cuda=args_train["use_cuda"],
                    train_hmm= args_classification["train_hmm"], save_weights=args_classification["save_hmm"], path_weights=args_classification["path_hmm"],
                    )
        else :
            hmm = HMM(loader.train_loader, hidden_dim = args_classification["hidden_state_hmm"],
                    nb_iter = args_classification["nb_iter_hmm"], nb_start = args_classification["nb_start_hmm"], use_cuda=args_train["use_cuda"],
                    train_hmm= args_classification["train_hmm"], save_weights=args_classification["save_hmm"], path_weights=args_classification["path_hmm"],
                    )
                    
        post_proc_regul = post_process_regularization(hmm, args_classification["nb_imputation"], use_cuda = args_train["use_cuda"])
        print("End Training HMM")
    elif args_classification["post_process_regularization"] is MICE_imputation_pretrained:
        import miceforest as mf
        import pandas as pd
        import numpy as np
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        Xtest, Ytest = next(iter(loader.test_loader))
        Xtest = Xtest.detach().numpy()
        df = pd.DataFrame(Xtest)
        df_amp = mf.ampute_data(df,perc=0.5,random_state=1991)
        imp = IterativeImputer(max_iter=10, random_state=0)
        X_test = df_amp.to_numpy()
        imp.fit(Xtest)
        post_proc_regul = post_process_regularization(network = imp, nb_imputation= args_classification["nb_imputation"])
    else :
        post_proc_regul = None

    return post_proc_regul


def get_networks(args_classification, args_destruction, args_complete_trainer, loader):
    input_size_destructor = args_destruction["input_size_destructor"]
    input_size_classifier = args_classification["input_size_classifier"]
    input_size_classifier_baseline = args_classification["input_size_classifier_baseline"]

    classifier =  args_classification["classifier"](input_size_classifier, loader.get_category())
    destructor = args_destruction["destructor"](input_size_destructor)
    

    if args_destruction["destructor_var"] is not None :
        destructor_var = args_destruction["destructor_var"](input_size_destructor)
    else :
        destructor_var = None

    
    if args_classification["classifier_baseline"] is not None :
        classifier_baseline = args_classification["classifier_baseline"](input_size_classifier_baseline,loader.get_category())
    else :
        classifier_baseline = None
 

    if args_complete_trainer["feature_extractor"] is not None :
        feature_extractor = args_complete_trainer["feature_extractor"]().cuda()
    else :
        feature_extractor = None


    return classifier, destructor, classifier_baseline, destructor_var, feature_extractor


def check_parameters_compatibility(args_classification, args_destruction, args_complete_trainer, args_train, args_test, args_output):
    sampling_distrib = args_train["sampling_distribution_train"]
    activation = args_destruction["activation"]
    # if args_train["sampling_distribution_train"] in [RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2X_Distribution_STE, L2X_Distribution] \
    #     and args_destruction["activation"] != torch.nn.LogSoftmax() :
    #     raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")
    
    # if args_train["sampling_distribution_train"] in [RelaxedBernoulli_thresholded_STE, RelaxedBernoulli] \
    #     and args_destruction["activation"] != torch.nn.LogSigmoid() :
    #     raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")


def experiment(args_dataset, args_classification, args_destruction, args_complete_trainer, args_train, args_test, args_output):
    
    dataset = args_dataset["dataset"]
    dic_list = {}
    ### Prepare output path :
    origin_path = args_output["path"]
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)
    folder = os.path.join(origin_path,dataset.__name__)
    if not os.path.exists(folder):
        os.makedirs(folder)

    experiment_name = args_output["experiment_name"]
    final_path = os.path.join(folder, experiment_name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    
    print(f"Save at {final_path}")
 
    save_parameters(final_path, args_dataset, args_classification,
                     args_destruction, args_complete_trainer,
                      args_train, args_test, args_output)

    check_parameters_compatibility(args_classification, args_destruction, args_complete_trainer, args_train, args_test, args_output)

    ### Datasets :
 
    
    dataset, loader = get_dataset(args_dataset, args_train, args_classification, args_destruction)
        


    kernel_patch = args_destruction["kernel_patch"]
    stride_patch = args_destruction["stride_patch"]
    

    ## Sampling :
    sampling_distribution_train = args_train["sampling_distribution_train"]
    sampling_distribution_train_var = args_train["sampling_distribution_train_var"]
    sampling_distribution_test = args_test["sampling_distribution_test"]
    use_cuda = args_train["use_cuda"]


    ### Imputation :
    imputationMethod = get_imputation_method(args_classification)  

    ### Multiple imputation :
    post_proc_regul = get_multiple_imputation(args_classification, args_train, loader)


    ### Networks :
    classifier, destructor, classifier_baseline, destructor_var, feature_extractor = get_networks(args_classification, args_destruction, args_complete_trainer, loader)



    ##### ============ Modules initialisation for ordinary training ============:

    if args_complete_trainer["complete_trainer"] is ordinaryTraining :

        scheduler_feature_extractor = None
        if args_complete_trainer["feature_extractor"] is not None :
            optim_feature_extractor = args_complete_trainer["feature_extractor"]()
            if args_train["scheduler_feature_extractor"] is not None :
                scheduler_feature_extractor = args_train["scheduler_feature_extractor"](optim_feature_extractor)
        else :
            optim_feature_extractor = None
            
        vanilla_classification_module = ClassificationModule(classifier, use_cuda = args_train["use_cuda"])
        vanilla_classification_module.kernel_update(kernel_patch, stride_patch)
        optim_classifier = args_train["optim_classification"](vanilla_classification_module.parameters())
        if args_train["scheduler_classification"] is not None :
            scheduler_classification = args_train["scheduler_classification"](optim_classifier)

        trainer_var = args_complete_trainer["complete_trainer"](vanilla_classification_module, feature_extractor=feature_extractor )
        nb_epoch = args_train["nb_epoch"]


        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(nb_epoch):
            dic_train = trainer_var.train_epoch(epoch, loader, optim_classifier, optim_feature_extractor= optim_feature_extractor,
                                                save_dic = True, print_dic_bool= ((epoch+1) % args_train["print_every"] == 0),
                                                scheduler_feature_extractor = scheduler_feature_extractor, scheduler_classification = scheduler_classification,
                                                )
            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch == nb_epoch-1:
                dic_test = trainer_var.test(loader)

            
            total_dic_train = fill_dic(total_dic_train, dic_train)
            total_dic_test = fill_dic(total_dic_test, dic_test)
            
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test"), total_dic_test)

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test

        return final_path, trainer_var, loader, dic_list


    if args_complete_trainer["complete_trainer"] is trainingWithSelection :

        
        scheduler_feature_extractor = None
        if args_complete_trainer["feature_extractor"] is not None :
            optim_feature_extractor = args_complete_trainer["feature_extractor"]()
            if args_train["scheduler_feature_extractor"] is not None :
                scheduler_feature_extractor = args_train["scheduler_feature_extractor"](optim_feature_extractor)
        else :
            optim_feature_extractor = None

        imputation = imputationMethod(input_size= args_classification["input_size_classification_module"], post_process_regularization = post_proc_regul,
                        reconstruction_reg= None, use_cuda = args_train["use_cuda"])
        vanilla_classification_module = ClassificationModule(classifier, use_cuda = args_train["use_cuda"],  feature_extractor=feature_extractor, imputation = imputation)
        vanilla_classification_module.kernel_update(kernel_patch, stride_patch)
        optim_classifier = args_train["optim_classification"](vanilla_classification_module.parameters())
        if args_train["scheduler_classification"] is not None :
            scheduler_classification = args_train["scheduler_classification"](optim_classifier)


        trainer_var = args_complete_trainer["complete_trainer"](vanilla_classification_module)
        nb_epoch = args_train["nb_epoch"]


        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(nb_epoch):
            dic_train = trainer_var.train_epoch(epoch, loader, optim_classifier, optim_feature_extractor= optim_feature_extractor,
                                                save_dic = True, print_dic_bool= ((epoch+1) % args_train["print_every"] == 0),
                                                scheduler_feature_extractor = scheduler_feature_extractor, scheduler_classification = scheduler_classification,
                                                )
            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch == nb_epoch-1:
                dic_test = trainer_var.test(loader)

            total_dic_train = fill_dic(total_dic_train, dic_train)
            total_dic_test = fill_dic(total_dic_test, dic_test)
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test"), total_dic_test)

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test

        return final_path, trainer_var, loader, dic_list


    
    if args_train["nb_epoch_pretrain"]>0 :

        scheduler_feature_extractor = None
        if args_complete_trainer["feature_extractor"] is not None :
            optim_feature_extractor = args_complete_trainer["feature_extractor"]()
            if args_train["scheduler_feature_extractor"] is not None :
                scheduler_feature_extractor = args_train["scheduler_feature_extractor"](optim_feature_extractor)
        else :
            optim_feature_extractor = None


        vanilla_classification_module = ClassificationModule(classifier, use_cuda = args_train["use_cuda"])
        optim_classifier = args_train["optim_classification"](vanilla_classification_module.parameters())
        if args_train["scheduler_classification"] is not None :
            scheduler_classification = args_train["scheduler_classification"](optim_classifier)

        vanilla_classification_module.kernel_update(kernel_patch, stride_patch)

        trainer_var = ordinaryTraining(vanilla_classification_module, feature_extractor=feature_extractor,)

        nb_epoch = args_train["nb_epoch_pretrain"]

        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(nb_epoch):
            dic_train = trainer_var.train_epoch(epoch, loader, optim_classifier, optim_feature_extractor= optim_feature_extractor,
                                                save_dic = True, print_dic_bool= ((epoch+1) % args_train["print_every"] == 0),
                                                scheduler_classification=scheduler_classification, scheduler_feature_extractor=scheduler_feature_extractor,
                                                )
            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch == nb_epoch-1:
                dic_test = trainer_var.test(loader)
        
            total_dic_train = fill_dic(total_dic_train, dic_train)
            total_dic_test = fill_dic(total_dic_test, dic_test)
            
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test"), total_dic_test)

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test


    ##### ============  Modules initialisation for complete training ===========:
    if args_complete_trainer["complete_trainer"] is not ordinaryTraining and args_complete_trainer["complete_trainer"] is not trainingWithSelection :
        
        if args_classification["reconstruction_regularization"] is not None :
            recons_regul = args_classification["reconstruction_regularization"](args_classification["autoencoder"], to_train = args_classification["train_reconstruction_regularization"])
        else : 
            recons_regul = None


        destruction_module = DestructionModule(destructor, feature_extractor=feature_extractor, activation=args_destruction["activation"], regularization=args_destruction["regularization"], use_cuda = args_train["use_cuda"])
        destruction_module.kernel_update(kernel_patch, stride_patch)
        if args_destruction["destructor_var"] is not None :
            destruction_module_var = DestructionModule(destructor_var, feature_extractor=feature_extractor, activation=args_destruction["activation"], regularization= free_regularization)
            destruction_module_var.kernel_update(kernel_patch, stride_patch)
        else :
            destruction_module_var = None
        imputation = imputationMethod(input_size= args_classification["input_size_classification_module"], post_process_regularization = post_proc_regul,
                        reconstruction_reg= recons_regul, use_cuda = args_train["use_cuda"])
        classification_module = ClassificationModule(classifier, imputation=imputation, feature_extractor=feature_extractor)
        classification_module.kernel_update(kernel_patch, stride_patch)
        

    



        if args_complete_trainer["complete_trainer"] is variationalTraining :
            trainer_var = args_complete_trainer["complete_trainer"](
                classification_module,
                destruction_module,
                destruction_module_var,
                baseline=classifier_baseline,
                feature_extractor=feature_extractor,
                use_cuda = use_cuda,
            )
        else :
            trainer_var = args_complete_trainer["complete_trainer"](
                classification_module,
                destruction_module,
                baseline=classifier_baseline,
                feature_extractor=feature_extractor,
                use_cuda = use_cuda,
                fix_classifier_parameters = args_train["fix_classifier_parameters"],
                post_hoc = args_train["post_hoc"]
            )


        ####Optimizer :
        optim_classification = args_train["optim_classification"](classification_module.parameters(), weight_decay = 1e-5)
        scheduler_classification = args_train["scheduler_classification"](optim_classification)

        optim_destruction = args_train["optim_destruction"](destruction_module.parameters(), weight_decay = 1e-5)
        scheduler_destruction = args_train["scheduler_destruction"](optim_destruction)


        if args_destruction["destructor_var"] is not None :
            optim_destruction_var = args_train["optim_destruction_var"](destruction_module_var.parameters(), weight_decay = 1e-5)
            scheduler_destruction_var = args_train["scheduler_destruction_var"](optim_destruction_var)
        else :
            optim_destruction_var = None
            scheduler_destruction_var = None

        
        if args_classification["classifier_baseline"] is not None :
            optim_baseline = args_train["optim_baseline"](classifier_baseline.parameters(), weight_decay = 1e-5)
            scheduler_baseline = args_train["scheduler_baseline"](optim_baseline)
        else :
            optim_baseline = None
            scheduler_baseline = None

    

        if args_complete_trainer["feature_extractor"] is not None :
            optim_feature_extractor = args_train["optim_feature_extractor"](feature_extractor.parameters(), weight_decay = 1e-5)
            scheduler_feature_extractor = args_train["scheduler_feature_extractor"](optim_feature_extractor)
        else :
            optim_feature_extractor = None
            scheduler_feature_extractor = None




######============== Complete module training================:

        temperature = torch.tensor([args_train["temperature_train_init"]])

        if torch.cuda.is_available():
            temperature = temperature.cuda()

        total_dic_train = {}
        total_dic_test_no_var = {}
        total_dic_test_no_var_2 = {}
        total_dic_test_var = {}
        for epoch in range(args_train["nb_epoch"]):
            current_sampling = get_distribution(sampling_distribution_train, temperature, args_train)
            current_sampling_test = get_distribution(sampling_distribution_test, temperature, args_train)


            if args_complete_trainer["complete_trainer"] is variationalTraining :
                current_sampling_var = get_distribution(sampling_distribution_train_var, temperature, args_train)
                dic_train = trainer_var.train_epoch( 
                    epoch, loader,
                    optim_classification, optim_destruction, optim_destruction_var,
                    current_sampling,
                    current_sampling_var,
                    optim_baseline=optim_baseline,
                    optim_feature_extractor= optim_feature_extractor,
                    lambda_reg = args_destruction["lambda_regularisation"],
                    lambda_reg_var = args_destruction["lambda_regularisation_var"],
                    lambda_reconstruction = args_classification["lambda_reconstruction"],
                    save_dic = True,
                    Nexpectation=args_train["Nexpectation_train"],
                    print_dic_bool = ((epoch+1) % args_train["print_every"] == 0),
                    scheduler_classification = scheduler_classification,
                    scheduler_destruction = scheduler_destruction,
                    scheduler_destruction_var = scheduler_destruction_var,
                    scheduler_baseline = scheduler_baseline, 
                    scheduler_feature_extractor = scheduler_feature_extractor,
                )
                if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch ==args_train["nb_epoch"]-1:
                    dic_test_no_var = trainer_var.test_no_var(loader, current_sampling_test, Nexpectation=args_test["Nexpectation_test"])
                    dic_test_var = trainer_var.test_var(loader, current_sampling_test, current_sampling_test, Nexpectation = args_test["Nexpectation_test"])
                    total_dic_test_var = fill_dic(total_dic_test_var, dic_test_var)
            else :   
                dic_train = trainer_var.train_epoch(
                    epoch, loader,
                    optim_classification, optim_destruction,
                    current_sampling,
                    optim_baseline=optim_baseline,
                    optim_feature_extractor= optim_feature_extractor,
                    lambda_reg = args_destruction["lambda_regularisation"],
                    lambda_reconstruction = args_classification["lambda_reconstruction"],
                    save_dic = True,
                    Nexpectation=args_train["Nexpectation_train"],
                    print_dic_bool = ((epoch+1) % args_train["print_every"] == 0),
                    scheduler_classification = scheduler_classification,
                    scheduler_destruction = scheduler_destruction,
                    scheduler_baseline = scheduler_baseline, 
                    scheduler_feature_extractor = scheduler_feature_extractor,
                )

                if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch ==args_train["nb_epoch"]-1:
                    dic_test_no_var = trainer_var.test_no_var(loader, current_sampling_test, Nexpectation=args_test["Nexpectation_test"])
                    total_dic_train = fill_dic(total_dic_train, dic_train)
                    total_dic_test_no_var = fill_dic(total_dic_test_no_var, dic_test_no_var)
           
            
            temperature *= args_train["temperature_decay"]
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test_no_var"), total_dic_test_no_var)
        if args_complete_trainer["complete_trainer"] is variationalTraining :
            save_dic(os.path.join(final_path, "test_var"), total_dic_test_var)
            dic_list["test_var"] = total_dic_test_var

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test_no_var
        

    return final_path, trainer_var, loader, dic_list 