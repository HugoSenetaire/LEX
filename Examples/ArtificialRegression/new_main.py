import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *
from interpretation_regression import *


from torch.distributions import *
from torch.optim import *
from functools import partial

if __name__ == '__main__' :


    args_output = {}
    args_output["path"] = "D:\\DTU\\firstProject\\MissingDataResults" # Path to results
    args_output["experiment_name"] = "no_name_no_critic"

    args_dataset = {}
    args_dataset["dataset"] = CircleDataset
    args_dataset["loader"] = LoaderArtificial

    args_classification = {}

    args_classification["input_size_classification_module"] = (1,2) # Size before imputation
    args_classification["input_size_classifier"] = (1,2) # Size after imputation
    args_classification["input_size_classifier_critic"] = (1,2) # Size before imputation (should be size of data)


    args_classification["classifier"] = StupidClassifier
    args_classification["classifier_critic"] = StupidClassifier


    args_classification["imputation"] = ConstantImputation
    args_classification["cste_imputation"] = 0
    args_classification["add_mask"] = False

    args_destruct = {}

    args_destruct["input_size_destructor"] = (1,2)
    args_destruct["input_size_autoencoder"] = (1,2)


  
    args_destruct["regularization"] = free_regularization
    args_destruct["lambda_regularisation"] = 0.1
    args_destruct["destructor"] = DestructorSimilar
    args_destruct["regularization_var"] = free_regularization
    args_destruct["lambda_regularization_var"] = 0.1
    args_destruct["destructor_var"] = None #DestructorSimilarVar

    args_destruct["autoencoder"] = AutoEncoder # Autoencoder Network to use
    args_destruct["post_process_regularization"] = None # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_destruct["reconstruction_regularization"] = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_destruct["lambda_reconstruction"] = 0.1 # Parameter for controlling the reconstruction regularization
    args_destruct["train_postprocess"] = False # If true, free the parameters of autoencoder during the training (loss guided by classification)
    args_destruct["train_reconstruction_regularization"] = False # If true, free the parameters of autoencoder during the training (loss guided by a reconstruction loss)
    args_destruct["noise_function"] = DropOutNoise(pi = 0.3) # Noise used to pretrain the autoencoder
    
    args_complete_trainer = {}
    args_complete_trainer["complete_trainer"] = noVariationalTraining_REINFORCE # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args_complete_trainer["feature_extractor"] = None

    args_train = {}
    args_train["nb_epoch"] = 1 # Training the complete model
    args_train["nb_epoch_pretrain_autoencoder"] = 10 # Training the complete model
    args_train["nb_epoch_pretrain"] = 0 # Training auto encoder
    args_train["Nexpectation_train"] = 10 # Number K in the IWAE-similar loss 

    args_train["sampling_distribution_train"] = Bernoulli # If using reparametrization (ie noVariationalTraining), need rsample
    args_train["sampling_distribution_train_var"] = Bernoulli
    args_train["temperature_train_init"] = 1.0
    args_train["temperature_decay"] = 0.5


    args_train["optim_classification"] = partial(Adam, lr=1e-4) #Learning rate for classification module
    args_train["optim_destruction"] = partial(Adam, lr=1e-4) # Learning rate for destruction module
    args_train["optim_destruction_var"] = partial(Adam, lr=1e-4) # Learning rate for the variationnal destruction module used in Variationnal Training
    args_train["optim_feature_extractor"] = partial(Adam, lr=1e-4) # Learning rate for the feature extractor if any
    args_train["optim_critic"] = partial(Adam, lr=1e-4) # Learning rate for the critic network
    args_train["optim_autoencoder"] = partial(Adam, lr=1e-4)

    
    args_test = {}
    args_test["sampling_distribution_test"] = Bernoulli # Sampling distribution used during test 
    args_test["temperature_test"] = 0.0
    args_test["Nexpectation_test"] = 10



    print("Start Experiment")
    final_path, trainer_var, loader = experiment(args_dataset,
                                        args_classification,
                                        args_destruct,
                                        args_complete_trainer,
                                        args_train, 
                                        args_test, 
                                        args_output)

    ## Interpretation

    data, target= next(iter(loader.test_loader))
    data = data[:200]
    target = target[:200]

    sampling_distribution_test = args_test["sampling_distribution_test"]
    
    if sampling_distribution_test is RelaxedBernoulli:
        current_sampling_test = partial(RelaxedBernoulli,args_test["temperature_test"])
    else :
        current_sampling_test = copy.deepcopy(sampling_distribution_test)
        
    pred = trainer_var._predict(data.cuda(), current_sampling_test, dataset = loader).detach().cpu().numpy()
    pred = np.argmax(pred, axis = 1)
    pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), current_sampling_test, 1)
    z_s = z_s.reshape(data.shape)

    data_destructed_pi, _ = trainer_var.classification_module.imputation.impute(data.cuda(), pi_list)
    data_destructed_pi = data_destructed_pi.detach().cpu().numpy()
    data_destructed_z, _ = trainer_var.classification_module.imputation.impute(data.cuda(), z_s)
    data_destructed_z = data_destructed_z.detach().cpu().numpy()



    save_result_artificial(final_path, data, target, pred)
    z_s = z_s.detach().cpu().numpy()
    pi_list = pi_list.detach().cpu().numpy()
    save_interpretation_artificial(final_path, data_destructed_pi, target, pred, prefix = "pi")
    save_interpretation_artificial(final_path, data_destructed_z, target, pred, prefix = "simple_z")
    save_interpretation_artificial_bar(final_path, pi_list, target, pred)


    data_expanded = get_extended_data(data, 100)
    data_expanded_flatten = data_expanded.flatten(0,1)


    pred = trainer_var._predict(data_expanded_flatten.cuda(), current_sampling_test, dataset = loader)
    
    pred = torch.logsumexp(pred.reshape(100, -1, loader.get_category()), 0).detach().cpu().numpy()
    pred = np.argmax(pred, axis = 1)
    pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), current_sampling_test, 100)
    z_s = z_s.reshape(data_expanded_flatten.shape)


    data_destructed_zs, _ = trainer_var.classification_module.imputation.impute(data_expanded_flatten.cuda(), z_s)
    data_destructed_zs = torch.mean(data_destructed_zs.reshape(100, -1, loader.get_category()),axis=0).detach().cpu().numpy()
    save_interpretation_artificial(final_path, data_destructed_zs, target, pred, prefix= "multi_z")