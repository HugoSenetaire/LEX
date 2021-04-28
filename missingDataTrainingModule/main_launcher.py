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




def get_imputation_method(args_class):
    
    if args_class["imputation"] is ConstantImputation:
        return partial(args_class["imputation"], cste = args_class["cste_imputation"], add_mask = args_class["add_mask"])
    elif args_class["imputation"] is LearnConstantImputation:
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])
    elif args_class["imputation"] is NoiseImputation :
        return partial(args_class["imputation"], add_mask = args_class["add_mask"])


def experiment(args_dataset, args_classification, args_destruction, args_complete_trainer, args_train, args_test, args_output):
    
    dataset = args_dataset["dataset"]
    
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

    ### Datasets :



    loader = args_dataset["loader"](dataset, root_dir = args_dataset["root_dir"], batch_size_test=args_dataset["batch_size_test"], batch_size_train=args_dataset["batch_size_train"])


    input_size_destructor = args_destruction["input_size_destructor"]
    input_size_autoencoder = args_destruction["input_size_autoencoder"]
    input_size_classifier = args_classification["input_size_classifier"]
    input_size_classifier_baseline = args_classification["input_size_classifier_baseline"]
    input_size_classification_module = args_classification["input_size_classification_module"]


    kernel_patch = args_destruction["kernel_patch"]
    stride_patch = args_destruction["stride_patch"]
    

    ## Sampling :
    sampling_distribution_train = args_train["sampling_distribution_train"]
    sampling_distribution_train_var = args_train["sampling_distribution_train_var"]
    sampling_distribution_test = args_test["sampling_distribution_test"]



    ### Imputation :
    imputationMethod = get_imputation_method(args_classification)  


    ### Autoencoder :

    noise_function = args_classification["noise_function"]
    train_reconstruction = args_classification["train_reconstruction_regularization"]
    train_postprocess = args_classification["train_postprocess"]
    reconstruction_regularization = args_classification["reconstruction_regularization"]
    post_process_regularization =  args_classification["post_process_regularization"]

    if args_classification["post_process_regularization"] is VAEAC_Imputation or args_classification["post_process_regularization"] is VAEAC_Imputation_DetachVersion :
        model, sampler = load_VAEAC(args_classification["VAEAC_dir"])
        post_proc_regul = post_process_regularization(model, sampler, args_classification["nb_imputation"])
    elif post_process_regularization is DatasetBasedImputation :
        post_proc_regul = post_process_regularization(loader.dataset, args_classification["nb_imputation"])
    elif args_classification["post_process_regularization"] is MICE_imputation :
        post_proc_regul = post_process_regularization(args_classification["nb_imputation"])
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



    ### Networks :


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

    ##### ============ Modules initialisation for ordinary training ============:

    if args_complete_trainer["complete_trainer"] is [postHocTraining, ordinaryTraining] or args_train["nb_epoch_pretrain"]>0 :
        if args_complete_trainer["feature_extractor"] is not None :
            optim_feature_extractor = args_complete_trainer["feature_extractor"]().cuda()
        else :
            optim_feature_extractor = None
        vanilla_classification_module = ClassificationModule(classifier)
        optim_classifier = args_train["optim_classification"](vanilla_classification_module.parameters())

        trainer_var = ordinaryTraining(vanilla_classification_module, feature_extractor=feature_extractor)
        for epoch in range(args_train["nb_epoch_pretrain"]):
            trainer_var.train_epoch(epoch, loader, optim_classifier, optim_feature_extractor= optim_feature_extractor)
            trainer_var.test(loader)
         
  

    ##### ============  Modules initialisation for complete training ===========:
    if args_complete_trainer["complete_trainer"] is not ordinaryTraining :
        
        if reconstruction_regularization is not None :
            recons_regul = reconstruction_regularization(autoencoder_network, to_train = train_reconstruction)
        else : 
            recons_regul = None


        destruction_module = DestructionModule(destructor, feature_extractor=feature_extractor, regularization=free_regularization)
        destruction_module.kernel_update(kernel_patch, stride_patch)
        if args_destruction["destructor_var"] is not None :
            destruction_module_var = DestructionModule(destructor_var, feature_extractor=feature_extractor, regularization= free_regularization)
            destruction_module_var.kernel_update(kernel_patch, stride_patch)
        else :
            destruction_module_var = None

        imputation = imputationMethod(input_size= input_size_classification_module,post_process_regularization = post_proc_regul,
                        reconstruction_reg= recons_regul)
        classification_module = ClassificationModule(classifier, imputation=imputation, feature_extractor=feature_extractor)
        classification_module.kernel_update(kernel_patch, stride_patch)
        

    



        if args_complete_trainer["complete_trainer"] is variationalTraining :
            trainer_var = args_complete_trainer["complete_trainer"](
                classification_module,
                destruction_module,
                destruction_module_var,
                baseline=classifier_baseline,
                feature_extractor=feature_extractor,
            )
        
        else :
            trainer_var = args_complete_trainer["complete_trainer"](
                classification_module,
                destruction_module,
                baseline=classifier_baseline,
                feature_extractor=feature_extractor,
            )


        ####Optimizer :
        optim_classification = args_train["optim_classification"](classification_module.parameters())
        optim_destruction = args_train["optim_destruction"](destruction_module.parameters())

        if args_destruction["destructor_var"] is not None :
            optim_destruction_var = args_train["optim_destruction_var"](destruction_module_var.parameters())
        else :
            optim_destruction_var = None

        
        if args_classification["classifier_baseline"] is not None :
            optim_baseline = args_train["optim_baseline"](classifier_baseline.parameters())
        else :
            optim_baseline = None
    

        if args_complete_trainer["feature_extractor"] is not None :
            optim_feature_extractor = args_train["optim_feature_extractor"](feature_extractor.parameters())
        else :
            optim_feature_extractor = None




######============== Complete module training================:

        temperature = torch.tensor([args_train["temperature_train_init"]])

        if torch.cuda.is_available():
            temperature = temperature.cuda()

        total_dic_train = {}
        total_dic_test_no_var = {}
        total_dic_test_no_var_2 = {}
        total_dic_test_var = {}
        for epoch in range(args_train["nb_epoch"]):
            current_sampling = get_distribution(sampling_distribution_train, temperature)
            print("Temperature", temperature)
            current_sampling_test = get_distribution(sampling_distribution_test, temperature)


            if args_complete_trainer["complete_trainer"] is variationalTraining :
                current_sampling_var = get_distribution(sampling_distribution_train_var, temperature)
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
                    Nexpectation=args_train["Nexpectation_train"]
                )
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
                    Nexpectation=args_train["Nexpectation_train"]
                )
                dic_test_no_var = trainer_var.test_no_var(loader, current_sampling_test, Nexpectation=args_test["Nexpectation_test"])

            total_dic_train = fill_dic(total_dic_train, dic_train)
            total_dic_test_no_var = fill_dic(total_dic_test_no_var, dic_test_no_var)
            
            temperature *= args_train["temperature_decay"]
        save_dic(os.path.join(final_path,"train"), total_dic_train)
        save_dic(os.path.join(final_path,"test_no_var"), total_dic_test_no_var)
        if args_complete_trainer["complete_trainer"] is variationalTraining :
            save_dic(os.path.join(final_path, "test_var"), total_dic_test_var)

    return final_path, trainer_var, loader