import sys

from torch import exp
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")
from missingDataTrainingModule import *
from datasets import *
from interpretation_regression import *
from default_parameter import *

from torch.distributions import *
from torch.optim import *
from functools import partial

if __name__ == '__main__' :
    args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler = get_default()


    args_dataset["dataset"] = HypercubeDataset
    args_classification["imputation"] = ConstantImputation
    args_classification["cste"] = 0
    args_classification["post_process_regularization"] = DatasetBasedImputation # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_dataset["give_index"] = True

    path_weights = os.path.join(args_output["path"], "2DDataset")
    name_experiment = "AllZTraining_MultipleImputation"



    args_complete_trainer["complete_trainer"] = AllZTraining
    args_train["sampling_distribution_train"] = Bernoulli
    args_test["sampling_distribution_test"] = Bernoulli
    args_destruction["activation"] = torch.nn.LogSigmoid()

    nb_experiments = 10


    regularization_module = ["L1Regularisation", "L2Regularization", "Rate05L2Regularization"]


    path_global = os.path.join(path_weights, name_experiment)

    list_classifier = {"ClassifierLVL3": ClassifierLVL3, "ClassifierLinear" :ClassifierLinear,}
    list_destructor = {"DestructorLVL3":DestructorLVL3,  }
    list_lambda = [0.0, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    for experiment_id in range(nb_experiments):
        for classifier_name in list_classifier :
            classifier = list_classifier[classifier_name]
            for destructor_name in list_destructor :
                destructor = list_destructor[destructor_name]
                for lambda_reg in list_lambda :
                    args_destruction["lambda_regularisation"] = lambda_reg
                    args_classification["classifier"] = classifier
                    args_destruction["destructor"] = destructor


                    
                    dic = {}
                    dic_count_batch_size = {}
                    dic_count_dim_batch_size = {}
                    args_dataset["centroids_path"] = os.path.join(path_weights, f"centroids_{experiment_id}.npy")
                    current_path = os.path.join(path_global, f"centroids_{experiment_id}")
                    current_path = os.path.join(current_path, f"{str(classifier_name)}_{str(destructor_name)}_{lambda_reg}")
                    args_output["path"] = current_path
                    print(f"Training the experiment id {experiment_id}")
                    final_path, trainer_var, loader, dic_list = experiment(args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = True)
                    if loader.dataset.nb_dim ==2:
                        plot_destructor_output(trainer_var.selection_module, loader.dataset, args_output["path"])
                        plot_destructor_output(trainer_var.selection_module, loader.dataset, args_output["path"], train_data=True)
                        plot_destructor_output(trainer_var.selection_module, loader.dataset, args_output["path"], interpretation= True)
                        plot_destructor_output(trainer_var.selection_module, loader.dataset, args_output["path"], interpretation= True, train_data=True)
                        # plot_model_output(trainer_var, loader.dataset, Bernoulli, args_output["path"])
                        # plot_model_output(trainer_var, loader.dataset, Bernoulli, args_output["path"], train_data=True)

                    # dic=get_dic_experiment(dic_list, dic)

                    ## Interpretation
                    sampling_distribution_test = args_test["sampling_distribution_test"]
                    current_sampling_test = get_distribution(sampling_distribution_test, temperature=args_test["temperature_test"], args_train = args_train)
                    # get_evaluation(trainer_var, loader, dic, dic_count_batch_size, dic_count_dim_batch_size, current_sampling_test, args_train, train = True)
                    # get_evaluation(trainer_var, loader, dic, dic_count_batch_size, dic_count_dim_batch_size, current_sampling_test, args_train, train = False)

                    print(f"Testing the experiment id {experiment_id}")

                

                    dic = normalize_dic(dic, dic_count_batch_size, dic_count_dim_batch_size)
                    with open(os.path.join(current_path,"results_dic.txt"), "w") as f:
                        f.write(str(dic))

                    
                    np.save(os.path.join(current_path, "results_dic.npy"), dic)
