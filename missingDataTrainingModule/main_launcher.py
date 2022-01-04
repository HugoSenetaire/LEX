# from .Destruction import * 
# from .Classification import *
# from .utils_missing import *
import numpy as np
from psutil import net_connections


from .classicTraining import *
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

def get_dataset(args_dataset,):
    dataset = args_dataset["dataset"](**args_dataset)
    loader = args_dataset["loader"](dataset, batch_size_test=args_dataset["batch_size_test"], batch_size_train=args_dataset["batch_size_train"],)
    return dataset, loader


def comply_size(dataset, args_classification, args_destruction, args_complete_trainer):
    dim_shape = dataset.get_dim_input()
    args_classification["input_size_classification_module"] = dim_shape # Size before imputation
    args_classification["input_size_classifier"] = dim_shape # Size after imputation
    args_classification["input_size_baseline"] = dim_shape # Size before imputation (should be size of data)
    args_destruction["input_size_destructor"] = dim_shape
    args_destruction["output_size_destructor"] = dim_shape
    args_destruction["input_size_autoencoder"] = dim_shape
    args_complete_trainer["input_size_baseline"] = dim_shape
    args_complete_trainer["reshape_mask_function"] = partial(args_complete_trainer["reshape_mask_function"], size = dim_shape)

def get_imputation_method(args_class):
    
    if args_class["imputation"] is ConstantImputation :
        return partial(args_class["imputation"], cste = args_class["cste_imputation"], add_mask = args_class["add_mask"])
    elif args_class["imputation"] is LearnConstantImputation:
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])
    elif args_class["imputation"] is NoiseImputation :
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])
    else :
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])


def get_multiple_imputation(args_classification, args_train, loader):
    post_process_regularization = args_classification["post_process_regularization"]
    if post_process_regularization is VAEAC_Imputation_DetachVersion :
        model, sampler = load_VAEAC(args_classification["VAEAC_dir"])
        post_proc_regul = post_process_regularization(model, sampler, args_classification["nb_imputation"])
    elif post_process_regularization is GaussianMixtureImputation :
        post_proc_regul = post_process_regularization(**args_classification)
    elif post_process_regularization is DatasetBasedImputation :
        post_proc_regul = post_process_regularization(loader.dataset, args_classification["nb_imputation"])
    elif post_process_regularization is MICE_imputation :
        post_proc_regul = post_process_regularization(args_classification["nb_imputation"])
    elif post_process_regularization is MarkovChainImputation :
        print("Training Markov Chain")
        markov_chain = MarkovChain(loader.train_loader, use_cuda=args_train["use_cuda"])
        post_proc_regul = post_process_regularization(markov_chain, args_classification["nb_imputation"], use_cuda=args_train["use_cuda"])
    elif post_process_regularization is HMMimputation:
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


def get_networks(args_classification, args_destruction, args_complete_trainer, output_category):
    input_size_destructor = args_destruction["input_size_destructor"]
    output_size_destructor = args_destruction["output_size_destructor"]
    input_size_classifier = args_classification["input_size_classification_module"]
    input_size_baseline = args_classification["input_size_classification_module"]

    classifier =  args_classification["classifier"](input_size_classifier, output_category)
    destructor = args_destruction["destructor"](input_size_destructor, output_size_destructor)
    

    if args_destruction["destructor_var"] is not None :
        destructor_var = args_destruction["destructor_var"](input_size_destructor, output_size_destructor)
    else :
        destructor_var = None

    
    if args_complete_trainer["baseline"] is not None :
        baseline = args_complete_trainer["baseline"](input_size_baseline, output_category)
    else :
        baseline = None
 
    return classifier, destructor, baseline, destructor_var

def get_regularization_method(args_destruction):
    regularization = args_destruction["regularization"](**args_destruction)
    return regularization



def check_parameters_compatibility(args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_output):
    sampling_distrib = args_distribution_module["distribution"]
    activation = args_destruction["activation"]
    if args_distribution_module["distribution"] in [RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2X_Distribution_STE, L2X_Distribution] \
        and args_destruction["activation"] != torch.nn.LogSoftmax() :
        raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")
    
    if args_distribution_module["distribution"] in [RelaxedBernoulli_thresholded_STE, RelaxedBernoulli] \
        and args_destruction["activation"] != torch.nn.LogSigmoid() :
        raise ValueError(f"Sampling distribution {sampling_distrib} is not compatible with the activation function {activation}")


def experiment(args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = True):
    
    dataset = args_dataset["dataset"]
    dic_list = {}

    ### Prepare output path :
    origin_path = args_output["path"]
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)

    if name_modification :
        print("Modification of the name")
        folder = os.path.join(origin_path,dataset.__name__)
        if not os.path.exists(folder):
            os.makedirs(folder)

        experiment_name = args_output["experiment_name"]
        final_path = os.path.join(folder, experiment_name)
    else :
        final_path = origin_path


    if not os.path.exists(final_path):
        os.makedirs(final_path)

    
    print(f"Save at {final_path}")
    save_parameters(final_path, args_dataset, args_classification,
                     args_destruction, args_complete_trainer,
                      args_train, args_test, args_output)
    check_parameters_compatibility(args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_output)

    ### Datasets :
    dataset, loader = get_dataset(args_dataset,)
    if args_complete_trainer["comply_with_dataset"] :
        comply_size(dataset, args_classification, args_destruction, args_complete_trainer)

    ## Sampling :
    distribution_module = get_distribution_module_from_args(args_distribution_module)

    ### Imputation :
    imputationMethod = get_imputation_method(args_classification)  

    ### Multiple imputation :
    post_proc_regul = get_multiple_imputation(args_classification, args_train, loader)

    ### Regularization method :
    regularization = get_regularization_method(args_destruction)

    ### Networks :
    classifier, destructor, baseline, destructor_var = get_networks(args_classification, args_destruction, args_complete_trainer, dataset.get_dim_output())

    post_hoc_guidance = None
    ##### ============ Training POST-HOC ============= ####

    if args_train["post_hoc"] and args_train["post_hoc_guidance"] is not None :
        print("Training post-hoc guidance")
        post_hoc_classifier =  args_train["post_hoc_guidance"](args_destruction["input_size_destructor"], dataset.get_dim_output())
        post_hoc_guidance = ClassificationModule(post_hoc_classifier, imputation = None)
        optim_post_hoc = args_train["optim_post_hoc"](post_hoc_guidance.parameters())

        if args_train["scheduler_post_hoc"] is not None :
            scheduler_post_hoc = args_train["scheduler_post_hoc"](optim_post_hoc)
        else :
            scheduler_post_hoc = None
        trainer = ordinaryTraining(classification_module = post_hoc_guidance,)
        if args_train["use_cuda"]:
            trainer.cuda()
        trainer.compile(optimizer=optim_post_hoc, scheduler_classification = scheduler_post_hoc)

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

    if args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trainingWithSelection  or args_complete_trainer["complete_trainer"] is trainingWithSelection or args_train["nb_epoch_pretrain"]>0 :
        
        if args_complete_trainer["complete_trainer"] is trainingWithSelection :
            imputation = imputationMethod(input_size= args_classification["input_size_classification_module"], post_process_regularization = post_proc_regul,
                        reconstruction_reg= None, )
        else :
            imputation = None
            
        if args_complete_trainer["complete_trainer"] is ordinaryTraining or args_complete_trainer["complete_trainer"] is trainingWithSelection :
            nb_epoch = args_train["nb_epoch"]
        else :
            nb_epoch = args_train["nb_epoch_pretrain"]

        vanilla_classification_module = ClassificationModule(classifier, imputation = imputation)
        if args_train["use_cuda"]:
            vanilla_classification_module.cuda()
        optim_classifier = args_train["optim_classification"](vanilla_classification_module.parameters())


        if args_train["scheduler_classification"] is not None :
            scheduler_classification = args_train["scheduler_classification"](optim_classifier)
        else :
            scheduler_classification = None

        trainer = args_complete_trainer["complete_trainer"](vanilla_classification_module,)
        trainer.compile(optimizer=optim_classifier, scheduler_classification = scheduler_classification,)


        total_dic_train = {}
        total_dic_test = {}
        for epoch in range(nb_epoch):
            dic_train = trainer.train_epoch(epoch, loader, save_dic = True, print_dic_bool= ((epoch+1) % args_train["print_every"] == 0),)
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



    ##### ============  Modules initialisation for complete training ===========:
    if args_complete_trainer["complete_trainer"] is not ordinaryTraining and args_complete_trainer["complete_trainer"] is not trainingWithSelection :

        if args_classification["reconstruction_regularization"] is not None :
            recons_regul = args_classification["reconstruction_regularization"](args_classification["autoencoder"], to_train = args_classification["train_reconstruction_regularization"])
        else : 
            recons_regul = None

        destruction_module = DestructionModule(destructor,
                            activation=args_destruction["activation"],
                            regularization=regularization,
                            )

        imputation = imputationMethod(input_size= args_classification["input_size_classification_module"],
                                    post_process_regularization = post_proc_regul,
                                    reconstruction_reg= recons_regul,
                                    )

        classification_module = ClassificationModule(classifier, imputation=imputation)
        

        if args_train["post_hoc"] and args_train["post_hoc_guidance"] is None :
            print("Training in PostHoc without a variational approximation")
            post_hoc_guidance = classification_module


        trainer = args_complete_trainer["complete_trainer"](
            classification_module,
            destruction_module,
            distribution_module = distribution_module,
            baseline = baseline,
            reshape_mask_function = args_complete_trainer["reshape_mask_function"],
            fix_classifier_parameters = args_train["fix_classifier_parameters"],
            post_hoc_guidance = post_hoc_guidance,
            argmax_post_hoc = args_train["argmax_post_hoc"],
        )

        if args_train["use_cuda"]:
            trainer.cuda()

        ####Optimizer :
        optim_classification = args_compiler["optim_classification"](classification_module.parameters(), weight_decay = 1e-5)
        if args_compiler["scheduler_classification"] is not None :
            scheduler_classification = args_compiler["scheduler_classification"](optim_classification)
        else :
            scheduler_classification = None


        optim_destruction = args_compiler["optim_destruction"](destruction_module.parameters(), weight_decay = 1e-5)
        if args_compiler["scheduler_destruction"] is not None :
            scheduler_destruction = args_compiler["scheduler_destruction"](optim_destruction)
        else :
            scheduler_destruction = None
        
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
            optim_destruction = optim_destruction,
            scheduler_classification = scheduler_classification,
            scheduler_destruction = scheduler_destruction,
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
            print(args_train["nb_sample_z_train"])
            dic_train = trainer.train_epoch(
                epoch, loader,
                save_dic = True,
                nb_sample_z=args_train["nb_sample_z_train"],
                verbose = ((epoch+1) % args_train["print_every"] == 0),
            )

            if (epoch+1)%args_complete_trainer["save_every_epoch"] == 0 or epoch ==args_train["nb_epoch"]-1:
                dic_test = trainer.test(loader, nb_sample_z=args_test["nb_sample_z_test"])
                total_dic_train = fill_dic(total_dic_train, dic_train)
                total_dic_test = fill_dic(total_dic_test, dic_test)
           
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test"), total_dic_test)
        if args_complete_trainer["complete_trainer"] is variationalTraining :
            save_dic(os.path.join(final_path, "test_var"), total_dic_test_var)
            dic_list["test_var"] = total_dic_test_var

        dic_list["train"] = total_dic_train
        dic_list["test"]  = total_dic_test
        

    return final_path, trainer, loader, dic_list 