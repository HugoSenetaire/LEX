import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *
from interpretation_image import *


from torch.distributions import *
from torch.optim import *
from functools import partial
from lime import lime_image
from skimage.segmentation import mark_boundaries
if __name__ == '__main__' :
    args_output = {}
    args_output["path"] = "D:\\DTU\\firstProject\\MissingDataResults" # Path to results
    args_output["experiment_name"] = "Relax_VAEAC_newly_trained_lambda10"

    args_dataset = {}
    args_dataset["dataset"] = FashionMNISTDataset
    args_dataset["loader"] = LoaderEncapsulation
    args_dataset["root_dir"] = "/scratch/hhjs/"

    args_classification = {}

    args_classification["input_size_classification_module"] = (1, 28, 28) # Size before imputation
    args_classification["input_size_classifier"] = (1, 28, 28) # Size after imputation
    args_classification["input_size_classifier_baseline"] = (1, 28, 28) # Size before imputation (should be size of data)


    args_classification["classifier"] = ClassifierModel
    args_classification["classifier_baseline"] = None


    args_classification["imputation"] = ConstantImputation
    args_classification["cste_imputation"] = 0
    args_classification["add_mask"] = False


    args_classification["post_process_regularization"] = VAEAC_Imputation_DetachVersion # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
    args_classification["reconstruction_regularization"] = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
    args_classification["lambda_reconstruction"] = 0.1 # Parameter for controlling the reconstruction regularization
    args_classification["train_postprocess"] = False # If true, free the parameters of autoencoder during the training (loss guided by classification)
    args_classification["train_reconstruction_regularization"] = False # If true, free the parameters of autoencoder during the training (loss guided by a reconstruction loss)
   
    
    args_classification["autoencoder"] = AutoEncoder # Autoencoder Network to use
    args_classification["noise_function"] = DropOutNoise(pi = 0.3) # Noise used to pretrain the autoencoder


    args_classification["VAEAC_dir"] = "fashion_mnist_dropout05"
    args_classification["nb_imputation"] = 3




    args_destruct = {}

    args_destruct["input_size_destructor"] = (1, 28, 28)
    args_destruct["input_size_autoencoder"] = (1, 28, 28)


  
    args_destruct["regularization"] = free_regularization
    args_destruct["lambda_regularisation"] = 0.0
    args_destruct["destructor"] = DestructorSimilar
    args_destruct["regularization_var"] = free_regularization
    args_destruct["lambda_regularisation_var"] = 0.1
    args_destruct["destructor_var"] = None #DestructorSimilarVar

   
    
    
    args_complete_trainer = {}
    # args_complete_trainer["complete_trainer"] = noVariationalTraining_REINFORCE # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args_complete_trainer["complete_trainer"] = noVariationalTraining # Ordinary training, Variational Traininig, No Variational Training, post hoc...
    args_complete_trainer["feature_extractor"] = None

    args_train = {}
    args_train["nb_epoch"] = 5 # Training the complete model
    args_train["nb_epoch_pretrain_autoencoder"] = 1 # Training the complete model
    args_train["nb_epoch_pretrain"] = 0 # Training auto encoder
    args_train["Nexpectation_train"] = 3 # Number K in the IWAE-similar loss 

    args_train["sampling_distribution_train"] = RelaxedBernoulli # If using reparametrization (ie noVariationalTraining), need rsample
    args_train["sampling_distribution_train_var"] = RelaxedBernoulli
    args_train["temperature_train_init"] = 1.0
    args_train["temperature_decay"] = 0.5


    args_train["optim_classification"] = partial(Adam, lr=1e-4) #Learning rate for classification module
    args_train["optim_destruction"] = partial(Adam, lr=1e-4) # Learning rate for destruction module
    args_train["optim_destruction_var"] = partial(Adam, lr=1e-4) # Learning rate for the variationnal destruction module used in Variationnal Training
    args_train["optim_feature_extractor"] = partial(Adam, lr=1e-4) # Learning rate for the feature extractor if any
    args_train["optim_baseline"] = partial(Adam, lr=1e-4) # Learning rate for the baseline network
    args_train["optim_autoencoder"] = partial(Adam, lr=1e-4)
    
    args_test = {}
    args_test["sampling_distribution_test"] = Bernoulli # Sampling distribution used during test 
    args_test["temperature_test"] = 0.0
    args_test["Nexpectation_test"] = 2

    final_path, trainer_var, loader = experiment(args_dataset,
                                    args_classification,
                                    args_destruct,
                                    args_complete_trainer,
                                    args_train, 
                                    args_test, 
                                    args_output)




    ## Interpretation

    data, target= next(iter(loader.test_loader))
    data = data[:20]
    target = target[:20]

    sampling_distribution_test = args_test["sampling_distribution_test"]
    if sampling_distribution_test is RelaxedBernoulli:
        current_sampling_test = partial(RelaxedBernoulli,args_test["temperature_test"])
    else :
        current_sampling_test = copy.deepcopy(sampling_distribution_test)
        
    sample_list, pred = trainer_var.MCMC(loader,data, target, current_sampling_test,5000, return_pred=True)
    save_interpretation(final_path,sample_list, data, target, suffix = "no_var",
                        y_hat = torch.exp(pred).detach().cpu().numpy(),
                        class_names=[str(i) for i in range(10)])


    pred = trainer_var._predict(data.cuda(), current_sampling_test, dataset = loader)
    image_output, _ = trainer_var._get_pi(data.cuda())
    image_output = image_output.detach().cpu().numpy()
    save_interpretation(final_path, image_output, data, target, suffix = "direct_destruction",
                        y_hat= torch.exp(pred).detach().cpu().numpy(),
                        class_names= [str(i) for i in range(10)])

