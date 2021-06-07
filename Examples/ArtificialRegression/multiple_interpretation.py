import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
sys.path.append("/home/hhjs/MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *
from interpretation_regression import *
from default_parameter import *

from torch.distributions import *
from torch.optim import *
from functools import partial




if __name__ == '__main__' :

    args_output, args_dataset, args_classification, args_destruct, args_complete_trainer, args_train, args_test = get_default()
    
    for cste in [0]:
        for lambda_reg in [0, 0.1, 1.0]:
            for name, post_process_regularization in [("Vanilla", None)]:
                for mask in [True, False]:
                    args_classification["add_mask"] = mask
                    if mask :
                        args_classification["input_size_classifier"] = (2,2) 
                    else :
                        args_classification["input_size_classifier"] = (1,2)


                    args_destruct["lambda_regularisation"] = lambda_reg
                    args_classification["cste_imputation"] = cste
                    
                    args_classification["post_process_regularization"] = post_process_regularization # Possibility NetworkTransform, Network add, NetworkTransformMask (the output of the autoencoder is given to classification)
                    
                    args_output["experiment_name"] ="Relax" +"_noise_imputation"+"_lambda_reg" + str(lambda_reg) + "_mask" + str(mask)
                    
                    args_classification["VAEAC_dir"] = "circle_dataset_model_studentsv2"
                    args_classification["nb_imputation"] = 3


                    print("Start Experiment")
                    final_path, trainer_var, loader = experiment(args_dataset,
                                                        args_classification,
                                                        args_destruct,
                                                        args_complete_trainer,
                                                        args_train, 
                                                        args_test, 
                                                        args_output)

                    ## Interpretation
                    trainer_var.train()
                    data, target= next(iter(loader.test_loader))
                    batch_size = 200
                    data = data[:batch_size]
                    target = target[:batch_size]
                    nb_imputation_multiple = trainer_var.classification_module.imputation.nb_imputation


                    sampling_distribution_test = args_test["sampling_distribution_test"]
                    
                    if sampling_distribution_test is RelaxedBernoulli:
                        current_sampling_test = partial(RelaxedBernoulli,args_test["temperature_test"])
                    else :
                        current_sampling_test = copy.deepcopy(sampling_distribution_test)
                        
                    pred = trainer_var._predict(data.cuda(), current_sampling_test, dataset = loader).detach().cpu().numpy()
                    pred = np.argmax(pred, axis = 1).reshape(nb_imputation_multiple, -1)[0]
                    pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), current_sampling_test, 1)
                    z_s = z_s.reshape(data.shape)

                    data_destructed_pi, _ = trainer_var.classification_module.imputation.impute(data.cuda(), pi_list)
                
                    data_destructed_pi = data_destructed_pi.reshape((nb_imputation_multiple, batch_size, -1))[0].detach().cpu().numpy()
                    data_destructed_z, _ = trainer_var.classification_module.imputation.impute(data.cuda(), z_s)
                    data_destructed_z = data_destructed_z.reshape((nb_imputation_multiple, batch_size, -1))[0].detach().cpu().numpy()



                    save_result_artificial(final_path, data, target, pred)
                    z_s = z_s.detach().cpu().numpy()
                    pi_list = pi_list.detach().cpu().numpy()

                    save_interpretation_artificial(final_path, data_destructed_pi, target, pred, prefix = "pi")
                    save_interpretation_artificial(final_path, data_destructed_z, target, pred, prefix = "simple_z")
                    save_interpretation_artificial_bar(final_path, pi_list, target, pred)

                    ### TO DELETE :

                    # if sampling_distribution_test is RelaxedBernoulli:
                    #     aux = Bernoulli
                    # else :
                    #     aux = partial(RelaxedBernoulli,args_test["temperature_test"])

                    # pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), aux, 1)
                    # z_s = z_s.reshape(data.shape)


                    # data_destructed_pi, _ = trainer_var.classification_module.imputation.impute(data.cuda(), pi_list)
                    # data_destructed_pi = data_destructed_pi.reshape((nb_imputation_multiple, batch_size, -1))[0].detach().cpu().numpy()
                    # data_destructed_z, _ = trainer_var.classification_module.imputation.impute(data.cuda(), z_s)
                    # data_destructed_z = data_destructed_z.reshape((nb_imputation_multiple, batch_size, -1))[0].detach().cpu().numpy()
                    # save_interpretation_artificial(final_path, data_destructed_pi, target, pred, prefix = "pi_Bern")
                    # save_interpretation_artificial(final_path, data_destructed_z, target, pred, prefix = "simple_z_Bern")


        #### END
                    data_expanded, data_expanded_flatten = get_extended_data(data, 100)
                    pred = trainer_var._predict(data_expanded_flatten.cuda(), current_sampling_test, dataset = loader)
                    pred = torch.logsumexp(pred.reshape(100, -1, loader.get_category()), 0).detach().cpu().numpy()
                    pred = np.argmax(pred, axis = 1)
                    pi_list, _, z_s, _ = trainer_var._destructive_test(data.cuda(), current_sampling_test, 100)
                    z_s = z_s.reshape(data_expanded_flatten.shape)

                    aux_imputation = copy.deepcopy(trainer_var.classification_module.imputation)
                    aux_imputation.add_mask = False
                    data_destructed_zs, _ = aux_imputation.impute(data_expanded_flatten.cuda(), z_s)
                    data_destructed_zs = torch.mean(data_destructed_zs.reshape(100, -1, data.shape[1]),axis=0).detach().cpu().numpy()
                    save_interpretation_artificial(final_path, data_destructed_zs, target, pred, prefix= "multi_z")