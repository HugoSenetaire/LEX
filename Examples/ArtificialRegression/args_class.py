from torch.nn import LogSoftmax, LogSigmoid

from datasets import args_dataset_parameters

class args_object():
    def __init__(self) -> None:
        pass

    def to_dic(self):
        return self.__dict__


class args_output():
    def __init__(self):
        self.path = None # Path to results
        self.save_weights = None
        self.experiment_name = None

class args_trainer():
    def __init__(self):
        self.complete_trainer = None
        self.monte_carlo_gradient_estimator = None
        self.save_every_epoch = None
        self.save_epoch_function = None
        self.baseline = None


class args_interpretable_module():
    def __init__(self):
        self.interpretable_module = None
        self.reshape_mask_function = None

class args_dataset():
    def __init__(self):
        self.dataset = None
        self.loader = None
        self.args_dataset_parameters = args_dataset_parameters()

class args_classification():
    def __init__(self):
        self.input_size_prediction_module = None
        self.classifier = None
        self.imputation = None
        self.cste_imputation = None
        self.sigma_noise_imputation = None
        self.add_mask = None
        self.module_imputation = None
        self.nb_imputation_iwae = None
        self.nb_imputation_iwae_test = None #If none is given, turn to 1
        self.nb_imputation_mc = None
        self.nb_imputation_mc_test = None #If none is given, turn to 1
        self.reconstruction_regularization = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
        self.network_reconstruction = None # Posssibility Autoencoder regularization (the output of the autoencoder is not given to classification, simple regularization of the mask)
        self.lambda_reconstruction = None # Parameter for controlling the reconstruction regularization
        self.post_process_regularization = None # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
        self.network_post_process = None # Autoencoder Network to use
        self.post_process_trainable = None # If true, pretrain the autoencoder with the training data
        self.mask_reg = None
        self.mask_reg_rate = None


class args_selection():
    def __init__(self):
        self.input_size_selector = None
        self.output_size_selector = None
        self.selector = None
        self.selector_var = None 
        self.activation = LogSigmoid()
        # self.activation = torch.nn.LogSoftmax(dim=-1)

        # For regularization :
        self.trainable_regularisation = None
        self.regularization = None
        self.lambda_reg = None 
        self.rate = None
        self.loss_regularization = None # L1, L2 
        self.batched = None


        self.regularization_var = None
        self.lambda_regularization_var = None
        self.rate_var = None
        self.loss_regularization_var = None
        self.batched_var = None

class args_distribution_module():
    def __init__(self):
        self.distribution_module = None
        self.distribution = None
        self.distribution_relaxed = None
        self.temperature_init = None
        self.test_temperature = None
        self.scheduler_parameter = None
        self.antitheis_sampling = None 

class args_classification_distribution_module():
    def __init__(self):
        self.classification_distribution_module = None
        self.classification_distribution = None
        self.classification_distribution_relaxed = None
        self.temperature_init = None
        self.test_temperature = None
        self.scheduler_parameter = None
        self.antitheis_sampling = None

class args_train():
    def __init__(self):
        self.nb_epoch = None # Training the complete model
        self.nb_epoch_post_hoc = None # Training post_hoc
        self.nb_epoch_pretrain_selector = None # Pretrain selector
        self.use_regularization_pretrain_selector = None # Use regularization when pretraining the selector
        self.nb_epoch_pretrain = None # Training the complete model 
        self.nb_sample_z_train_monte_carlo = None
        self.nb_sample_z_train_IWAE = None  # Number K in the IWAE-similar loss
        self.nb_sample_z_train_monte_carlo_classification = None
        self.nb_sample_z_train_IWAE_classification = None
        self.loss_function = None # NLL, MSE
        self.loss_function_selection = None # This is used for DECOUPLED SELECTION WHEN ONE WANTS A DIFFERENT LOSS THERE
        self.verbose = False

        self.training_type = None # Options are ["classic", "alternate_ordinary", "alternate_fixing"]
        self.nb_step_fixed_classifier = None # Options for alternate fixing (number of step with fixed classifier)
        self.nb_step_fixed_selector = None # Options for alternate fixing (number of step with fixed selector)
        self.nb_step_all_free = None # Options for alternate fixing (number of step with all free)
        self.ratio_class_selection = None # Options for alternate ordinary Ratio of training with only classification compared to selection
        self.print_every = None

        self.sampling_subset_size = None # Sampling size for the subset 
        self.use_cuda = None
        self.fix_classifier_parameters = None
        self.fix_selector_parameters = None
        self.post_hoc = None
        self.argmax_post_hoc = None
        self.post_hoc_guidance = None

class args_test():
    def __init__(self):
        self.nb_sample_z_mc_test = None
        self.nb_sample_z_iwae_test = None
        self.liste_mc = None

class args_compiler():
    def __init__(self):
        self.optim_classification = None #Learning rate for classification module
        self.optim_selection = None # Learning rate for selection module
        self.optim_selection_var = None # Learning rate for the variationnal selection module used in Variationnal Training
        self.optim_distribution_module = None # Learning rate for the feature extractor if any
        self.optim_baseline = None # Learning rate for the baseline network
        self.optim_autoencoder = None
        self.optim_post_hoc = None

        
        self.optim_classification_param = None 
        self.optim_selection_param = None
        self.optim_selection_var_param = None 
        self.optim_distribution_module_param =None 
        self.optim_baseline_param = None
        self.optim_autoencoder_param = None
        self.optim_post_hoc_param = None


        self.scheduler_classification = None #Learning rate for classification module
        self.scheduler_selection = None # Learning rate for selection module
        self.scheduler_selection_var = None # Learning rate for the variationnal selection module used in Variationnal Training
        self.scheduler_distribution_module = None # Learning rate for the feature extractor if any
        self.scheduler_baseline = None # Learning rate for the baseline network
        self.scheduler_autoencoder = None
        self.scheduler_post_hoc = None
               
        self.scheduler_classification_param = None 
        self.scheduler_selection_param = None 
        self.scheduler_selection_var_param = None 
        self.scheduler_distribution_module_param = None 
        self.scheduler_baseline_param = None 
        self.scheduler_autoencoder_param = None
        self.scheduler_post_hoc_param = None
    
        

    
class complete_args():
    def __init__(self):
        self.args_output = args_output()
        self.args_trainer = args_trainer()
        self.args_selection = args_selection()
        self.args_distribution_module = args_distribution_module()
        self.args_classification_distribution_module = args_classification_distribution_module()
        self.args_classification = args_classification()
        self.args_train = args_train()
        self.args_test = args_test()
        self.args_compiler = args_compiler()
        self.args_dataset = args_dataset()
        self.args_interpretable_module = args_interpretable_module()

    