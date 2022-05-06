from default_parameter import *

from missingDataTrainingModule import *
from datasets import *
from interpretation_protein import *
from default_parameter import *

from torch.distributions import *
from torch.optim import *
from functools import partial

def get_relax(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = noVariationalTraining
    args_classification["classifier_baseline"] = None
    args_train["sampling_distribution_train"] = RelaxedBernoulli

def get_reinforce(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = REINFORCE
    args_classification["classifier_baseline"] = None
    args_train["sampling_distribution_train"] = Bernoulli

def get_reinforce_baseline(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = REINFORCE
    args_classification["classifier_baseline"] = StupidClassifier
    args_train["sampling_distribution_train"] = Bernoulli

def get_all_z_baseline(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = AllZTraining
    args_classification["classifier_baseline"] = None
    args_train["sampling_distribution_train"] = Bernoulli




if __name__ == '__main__' :

    args_output, args_dataset, args_classification, args_destruct, args_complete_trainer, args_train, args_test = get_default()

    args_output["experiment_name"] = "Simple_Test_Protein"

  

    final_path, trainer_var, loader, dic_list = experiment(args_dataset,
                                        args_classification,
                                        args_destruct,
                                        args_complete_trainer,
                                        args_train, 
                                        args_test, 
                                        args_output)

    ## Interpretation

    data, targets= next(iter(loader.test_loader))
    data = data[:20]
    targets = targets[:20]

    sampling_distribution_test = args_test["sampling_distribution_test"]
    
    if sampling_distribution_test is RelaxedBernoulli:
        current_sampling_test = partial(RelaxedBernoulli,args_test["temperature_test"])
    else :
        current_sampling_test = copy.deepcopy(sampling_distribution_test)

        
   
    pred = torch.exp(trainer_var._predict(data.cuda(), current_sampling_test, dataset = loader)).detach().cpu().numpy()
    image_interpretation, _ = trainer_var._get_pi(data.cuda())
    image_interpretation = image_interpretation.detach().cpu().numpy()
    interpretation_protein(final_path, data.detach().cpu().numpy(), image_interpretation, targets, pred)
    
