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
    # list_dataset = {"SimpleMNIST":DatasetMnist(64,1000),"VariationMNIST":DatasetMnistVariation(64,1000)}
    # list_dataset = [DatasetMnist(64,1000),DatasetMnistVariation(64,1000)]
    list_dataset = [MnistVariation1]
    lambda_reg_list = [0.0, 0.001]

    lr_list = [1e-4]
    nb_epoch=3

    imputationMethod_list = [ConstantImputation(), ConstantImputation(cste = 2000), LearnConstantImputation(), ConstantImputation(add_mask=True)]
    # imputationMethod_list = [MaskConstantImputation()]


    path_save = "D:\DTU\ResultsTEST2"
    if not os.path.exists(path_save):
        os.makedirs(path_save)



   #============ No Variationnal ===============

    for dataset in list_dataset:
        folder = os.path.join(path_save,os.path.join("Linear no variationnal", type(dataset).__name__))
        for lr in lr_list :
            for lambda_reg in lambda_reg_list :
                # for lambda_reg_var in lambda_reg_list :
                    for imputation in imputationMethod_list :
                        # final_path = os.path.join(folder, str(datetime.now()).replace(" ","_").replace(".","").replace(":","-"))
                        mnist = LoaderEncapsulation(dataset)              
                        parameter = {
                            "lr":lr,
                            "lambda_reg":lambda_reg,
                            # "lambda_reg_var": lambda_reg_var,
                            "imputation": str(imputation),
                        }
                        to_add = ""
                        for key in parameter.keys():
                            to_add += f"{key}_{parameter[key]}_".split("imputation")[-1].replace(" ","_").replace(".",";").replace(":","-").replace("'","").replace(">","")
                        final_path = os.path.join(folder, to_add)
                        if not os.path.exists(final_path):
                            os.makedirs(final_path)

                        with open(os.path.join(final_path,"class.txt"), "w") as f:
                            f.write(str(parameter))

                        input_size_classifier = (1,28,28)
                        if imputation.add_mask :
                            input_size_classifier = (2,28,28)
                        input_size_destructor = (1,28,28)


                        destructor_no_var = Destructor(input_size_destructor)
                        # destructor_var = DestructorVariational(input_size_destructor)
                        destruction_var = DestructionModule(destructor_no_var,
                            regularization=free_regularization,
                        )

                        classifier_var = ClassifierModel(input_size_classifier, mnist.get_category())
                        # print(classifier_var)
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
                                save_dic = True
                            )
                            # print(dic_train)
                            # dic_train_trainer_var.train(k, mnist, optim_classification, optim_destruction, partial(RelaxedBernoulli,temperature) , partial(RelaxedBernoulli,temperature), lambda_reg=0.0)
                            dic_test_no_var = trainer_var.test_no_var(mnist, Bernoulli)
                            temperature *= 0.5

                            total_dic_train = fill_dic(total_dic_train, dic_train)
                            total_dic_test_no_var = fill_dic(total_dic_test_no_var, dic_test_no_var)

                        
                        save_dic(os.path.join(final_path,"train"), total_dic_train)
                        save_dic(os.path.join(final_path,"test_no_var"), total_dic_test_no_var)


                        data, target= next(iter(mnist.test_loader))
                        data = data[:2]
                        target = target[:2]
                        sample_list = trainer_var.MCMC(mnist,data, target, Bernoulli,5000)
                        save_interpretation(final_path,sample_list, data, target, suffix = "no_var")


        


    
