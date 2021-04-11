import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *
from interpretation_regression import *
from default_parameter import *

from torch.distributions import *
from torch.optim import *
from functools import partial

def get_relax(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = noVariationalTraining
    args_classification["classifier_baseline"] = None
    args_train["sampling_distribution_train"] = RelaxedBernoulli

def get_reinforce(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = noVariationalTraining_REINFORCE
    args_classification["classifier_baseline"] = None
    args_train["sampling_distribution_train"] = Bernoulli

def get_reinforce_baseline(args_complete_trainer, args_classification, args_train):
    args_complete_trainer["complete_trainer"] = noVariationalTraining_REINFORCE
    args_classification["classifier_baseline"] = StupidClassifier
    args_train["sampling_distribution_train"] = Bernoulli



if __name__ == '__main__' :

    args_output, args_dataset, args_classification, args_destruct, args_complete_trainer, args_train, args_test = get_default()
    for method in ["reinforce", "reinforce_baseline"]:
        if method == "reinforce_baseline":
            get_reinforce_baseline(args_complete_trainer, args_classification, args_train)
        if method == "relax":
            get_relax(args_complete_trainer, args_classification, args_train)
        if method == "reinforce":
            get_reinforce(args_complete_trainer, args_classification, args_train)
        for cste in [0, -1, 10]:
            for lambda_reg in [0, 0.01, 0.1, 1.0, 10]:
                args_destruct["lambda_regularisation"] = lambda_reg
                args_classification["cste_imputation"] = cste
                args_output["experiment_name"] = method + "_cste_" + str(cste) + "_lambda_reg_" + str(lambda_reg) 


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

                ### TO DELETE :

                if sampling_distribution_test is RelaxedBernoulli:
                    aux = Bernoulli
                else :
                    aux = partial(RelaxedBernoulli,args_test["temperature_test"])

                pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), aux, 1)
                z_s = z_s.reshape(data.shape)

                data_destructed_pi, _ = trainer_var.classification_module.imputation.impute(data.cuda(), pi_list)
                data_destructed_pi = data_destructed_pi.detach().cpu().numpy()
                data_destructed_z, _ = trainer_var.classification_module.imputation.impute(data.cuda(), z_s)
                data_destructed_z = data_destructed_z.detach().cpu().numpy()
                save_interpretation_artificial(final_path, data_destructed_pi, target, pred, prefix = "pi_Bern")
                save_interpretation_artificial(final_path, data_destructed_z, target, pred, prefix = "simple_z_Bern")


                #### END
                # data_expanded, data_expanded_flatten = get_extended_data(data, 100)
                # pred = trainer_var._predict(data_expanded_flatten.cuda(), current_sampling_test, dataset = loader)
                # pred = torch.logsumexp(pred.reshape(100, -1, loader.get_category()), 0).detach().cpu().numpy()
                # pred = np.argmax(pred, axis = 1)
                # pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), current_sampling_test, 100)
                # z_s = z_s.reshape(data_expanded_flatten.shape)

                # aux_imputation = copy.deepcopy(trainer_var.classification_module.imputation)
                # aux_imputation.add_mask = False
                # data_destructed_zs, _ = aux_imputation.impute(data_expanded_flatten.cuda(), z_s)
                # data_destructed_zs = torch.mean(data_destructed_zs.reshape(100, -1, data.shape[1]),axis=0).detach().cpu().numpy()
                # save_interpretation_artificial(final_path, data_destructed_zs, target, pred, prefix= "multi_z")