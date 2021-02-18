from datasets import *
from missingDataTrainingModule import *
from utils import *

from torch.distributions import *
from torch.optim import *

import os
from datetime import datetime
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt





if __name__ == "__main__":
    list_dataset = [MnistVariationFashion]
    lambda_reg_list = [0.1]

    lr = 1e-4
    nb_epoch=5
    nb_epoch_pretrain = [1]

    list_noiseFunction = [
        GaussianNoise(sigma=1.0, regularize = True),
        # DropOutNoise(0.1)
        ]
    imputationMethod_list = [ConstantImputation]


    train_or_not_reconstruction = [False]
    train_or_not_postprocess = [False]
    # reconstruction_regularization_list = [AutoEncoderReconstructionRegularization] 
    # post_process_regularization_list = [None]

    reconstruction_regularization_list = [None] 
    # post_process_regularization_list = [NetworkTransform]
    post_process_regularization_list = [NetworkAdd]


    path_save = "D:\DTU\SimpleTestRun"
    if not os.path.exists(path_save):
        os.makedirs(path_save)



   #============ No Variationnal ===============
    i = 2
    for dataset in list_dataset:
        folder = os.path.join(path_save,os.path.join("Linear no variationnal Autoencoder", dataset.__name__))
        for train_reconstruction in train_or_not_reconstruction :
            for train_postprocess in train_or_not_postprocess :
                for lambda_reg in lambda_reg_list :
                    for epoch_pretrain in nb_epoch_pretrain :
                        if epoch_pretrain==0 :
                            noise_list = [DropOutNoise(pi=0.)]
                        else :
                            noise_list = list_noiseFunction


                        for noise_function in noise_list :

                            input_size_autoencoder = (1,28,28)
                            input_size_destructor = (1,28,28)


                            #### AUTOENCODER TRAINING

                            
                            mnist = LoaderEncapsulation(dataset)

                            autoencoder_network = AutoEncoder(input_size=input_size_autoencoder).cuda()

                            mnist_noise = LoaderEncapsulation(dataset, noisy=True, noise_function=noise_function)
                            optim_autoencoder = Adam(autoencoder_network.parameters())
                            data_autoencoder, target_autoencoder = next(iter(mnist_noise.test_loader))
                            data_autoencoder = data_autoencoder[:4]
                            target_autoencoder = target_autoencoder[:4]
                            
                           
                                
                            for epoch in range(epoch_pretrain):
                                train_autoencoder(autoencoder_network, mnist_noise, optim_autoencoder)
                                test_autoencoder(autoencoder_network, mnist_noise)
                                autoencoder_network.eval()

                        

                            for reconstruction_regularization in reconstruction_regularization_list :
                                for post_process_regularization in post_process_regularization_list :          
                                    parameter = {
                                        "train_reconstruction": train_reconstruction,
                                        "train_postprocess" : train_postprocess,
                                        "recons_reg" : reconstruction_regularization,
                                        "post_process_reg" : post_process_regularization,
                                        "epoch_pretrain": epoch_pretrain,
                                        "noise_function":noise_function,
                                        "lambda_reg":lambda_reg,
                                        "post_process_regularization":post_process_regularization,
                                        "reconstruction_regularization":reconstruction_regularization
                                    }
                                    print(parameter)
                                    i+=1
                                    to_add = f"experiment{i}"
                                    # for key in parameter.keys():
                                    #     to_add += f"{key}_{parameter[key]}_".split("imputation")[-1].replace(" ","_").replace(".",";").replace(":","-").replace("'","").replace(">","")
                                    final_path = os.path.join(folder, to_add)
                                    if not os.path.exists(final_path):
                                        os.makedirs(final_path)

                                    with open(os.path.join(final_path,"class.txt"), "w") as f:
                                        f.write(str(parameter))

                                    save_interpretation(final_path,
                                    data_autoencoder.detach().cpu().numpy(), 
                                    target_autoencoder.detach().cpu().numpy(), [0,1,2,3],prefix= "input_autoencoder")
                        
                                    autoencoder_network_missing = copy.deepcopy(autoencoder_network)
                                    output = autoencoder_network_missing(data_autoencoder.cuda()).reshape(data_autoencoder.shape)

                                    save_interpretation(final_path,
                                    output.detach().cpu().numpy(), 
                                    target_autoencoder.detach().cpu().numpy(), [0,1,2,3], prefix="output_autoencoder_before_training")


                                    ##### Missing Data destruc training
                                    destructor_no_var = Destructor(input_size_destructor)
                                    destruction_var = DestructionModule(destructor_no_var,
                                        regularization=free_regularization,
                                    )
                                    if post_process_regularization is not None :
                                        post_proc_regul = post_process_regularization(autoencoder_network_missing, to_train = train_postprocess)
                                    else :
                                        post_proc_regul = None
                                    
                                    if reconstruction_regularization is not None :
                                        recons_regul = reconstruction_regularization(autoencoder_network_missing, to_train = train_reconstruction)
                                    else : recons_regul = None
                                    if post_process_regularization is NetworkAdd :
                                        input_size_classifier = (2,28,28)
                                    else :
                                        input_size_classifier = (1,28,28)
                                    input_size_classification_module = (1,28,28)
                                    

                                    # print("INPUT SIZE CLASSIFIER", input_size_classifier)
                                    imputation = ConstantImputation(input_size= input_size_classification_module,post_process_regularization = post_proc_regul,
                                                 reconstruction_reg= recons_regul)
                                    
                                    classifier_var = ClassifierModel(input_size_classifier, mnist.get_category())
                                    classification_var = ClassificationModule(classifier_var, imputation=imputation)

                                    trainer_var = noVariationalTraining(
                                        classification_var,
                                        destruction_var,
                                    )


                                    optim_classification = Adam(classification_var.parameters(), lr=lr)
                                    optim_destruction = Adam(destruction_var.parameters(), lr=lr)
                                    

                                    temperature = torch.tensor([1.0])

                                    if torch.cuda.is_available():
                                        temperature = temperature.cuda()

                                    total_dic_train = {}
                                    total_dic_test_no_var = {}
                                    for epoch in range(nb_epoch):
                                        
                                        dic_train = trainer_var.train(
                                            epoch, mnist,
                                            optim_classification, optim_destruction,
                                            partial(RelaxedBernoulli,temperature),
                                            lambda_reg=lambda_reg,
                                            lambda_reconstruction = 0.1,
                                            save_dic = True
                                        )
                                        dic_test_no_var = trainer_var.test_no_var(mnist, Bernoulli)
                                        temperature *= 0.5

                                        total_dic_train = fill_dic(total_dic_train, dic_train)
                                        total_dic_test_no_var = fill_dic(total_dic_test_no_var, dic_test_no_var)

                                    

                                    ###### Sample and save results
                                    save_dic(os.path.join(final_path,"train"), total_dic_train)
                                    save_dic(os.path.join(final_path,"test_no_var"), total_dic_test_no_var)


                                    data, target= next(iter(mnist.test_loader))
                                    data = data[:2]
                                    target = target[:2]
                                    sample_list, pred = trainer_var.MCMC(mnist,data, target, Bernoulli,5000, return_pred=True)

                                    save_interpretation(final_path,sample_list, data, target, suffix = "no_var",
                                         y_hat = torch.exp(pred).detach().cpu().numpy(),
                                         class_names=[str(i) for i in range(10)])

                                    output = autoencoder_network_missing(data_autoencoder.cuda()).reshape(data_autoencoder.shape)
                                    save_interpretation(final_path,
                                        output.detach().cpu().numpy(), 
                                        target_autoencoder.detach().cpu().numpy(), [0,1,2,3],
                                         prefix = "output_autoencoder_after_training",
                                         )




                


    
