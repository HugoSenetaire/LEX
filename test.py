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
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


if __name__ == "__main__":
    dataset_encapsulation = ReligionDataset()

    input_size_destructor = dataset_encapsulation.get_shape()
    input_size_autoencoder = dataset_encapsulation.get_shape()
    input_size_classifier = dataset_encapsulation.get_shape()
    input_size_classification_module = dataset_encapsulation.get_shape()

    output_size = 2

    lambda_reg = 1.0

    lr = 1e-4
    nb_epoch=30
    epoch_pretrain = 0

    noise_function = GaussianNoise(sigma=0.5, regularize=False)
    imputationMethod_list = ConstantImputation
    train_reconstruction = False
    train_postprocess = False

    reconstruction_regularization = None 
    post_process_regularization = None
    if reconstruction_regularization is None and post_process_regularization is None :
        need_autoencoder = False

    else :
        need_autoencoder = True


    path_save = "D:\DTU\Text"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    folder = os.path.join(path_save,"religion_dataset")
    if not os.path.exists(folder):
        os.makedirs(folder)
    experiment_name = "post_hoc_longer_training"
    folder = os.path.join(folder, experiment_name)
    if not os.path.exists(folder):
        os.makedirs(folder)


    #============ Autoencoder ===================

    if need_autoencoder :
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


    classifier = ClassifierModel(input_size=input_size_classifier, output= output_size, middle_size=1000)

    if need_autoencoder :
        # save_interpretation(folder,
        # data_autoencoder.detach().cpu().numpy(), 
        # target_autoencoder.detach().cpu().numpy(), [0,1,2,3],prefix= "input_autoencoder")

        autoencoder_network_missing = copy.deepcopy(autoencoder_network)
        output = autoencoder_network_missing(data_autoencoder.cuda()).reshape(data_autoencoder.shape)

        # save_interpretation(folder,
        # output.detach().cpu().numpy(), 
        # target_autoencoder.detach().cpu().numpy(), [0,1,2,3], prefix="output_autoencoder_before_training")
    classification_vanilla = ClassificationModule(classifier, imputation=None, imputation_reg=None)

    optim_classifier = Adam(classification_vanilla.parameters())
    trainer_vanilla = ordinaryTraining(classification_vanilla)

    for epoch in range(5):
        trainer_vanilla.train(epoch, dataset_encapsulation, optim_classifier)
        trainer_vanilla.test(dataset_encapsulation)
    
    classifier = classifier.cpu()
    pipeline = TextPipeline(
        [classifier, lambda x: torch.exp(x)], 
        list_prepare= [
            dataset_encapsulation.vectorizer.transform,
            scipy.sparse.csr_matrix.todense,
            lambda x: torch.tensor(x, dtype=torch.float32),
            ]
        )


    list_idx = [1,10,50,83]
    newsgroups_test = dataset_encapsulation.newsgroups_test

    for idx in list_idx :
        explainer = LimeTextExplainer(class_names=dataset_encapsulation.class_names)
        exp = explainer.explain_instance(newsgroups_test.data[idx], pipeline.predict)
        print('Document id: %d' % idx)
        print(newsgroups_test.data[idx])
        print('Probability(christian) =', pipeline.predict([newsgroups_test.data[idx]])[0, 1])
        print('True class: %s' % dataset_encapsulation.class_names[newsgroups_test.target[idx]])
        print(exp.as_list())

        exp.save_to_file(os.path.join(folder,f'test{idx}.html'))




    ## POST HOC TRAINING :
    destructor_no_var = Destructor(input_size_destructor)
    destruction_var = DestructionModule(destructor_no_var,
        regularization=free_regularization,
    )


    

    imputation = ConstantImputation(input_size= input_size_classification_module,post_process_regularization = post_process_regularization,
                    reconstruction_reg= reconstruction_regularization)
    
    classifier_var = copy.deepcopy(classifier)
    classification_var = ClassificationModule(classifier_var, imputation=imputation)

    trainer_var = postHocTraining(
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
            epoch, dataset_encapsulation,
            optim_classification, optim_destruction,
            partial(RelaxedBernoulli,temperature),
            lambda_reg=lambda_reg,
            lambda_reconstruction = 0.1,
            save_dic = True
        )
        dic_test_no_var = trainer_var.test_no_var(dataset_encapsulation, Bernoulli)
        temperature *= 0.5

        total_dic_train = fill_dic(total_dic_train, dic_train)
        total_dic_test_no_var = fill_dic(total_dic_test_no_var, dic_test_no_var)

    

    ###### Sample and save results
    save_dic(os.path.join(folder,"train"), total_dic_train)
    save_dic(os.path.join(folder,"test_no_var"), total_dic_test_no_var)


    explainer = LimeTextExplainer(class_names=dataset_encapsulation.class_names)
    pipeline = TextPipeline(
    [copy.deepcopy(classifier_var).cpu(), lambda x: torch.exp(x)], 
    list_prepare= [
        dataset_encapsulation.vectorizer.transform,
        scipy.sparse.csr_matrix.todense,
        lambda x: torch.tensor(x, dtype=torch.float32),
        ]
    )

    
    newsgroups_test = dataset_encapsulation.newsgroups_test

    for idx in list_idx :
        doc_to_explain = newsgroups_test.data[idx]
        data, target = dataset_encapsulation.dataset_test.__getitem__(idx)

        data = data.unsqueeze(0)
        target = target.unsqueeze(0)

        sample_list, pred = trainer_var.MCMC(dataset_encapsulation,data, target, Bernoulli,5000,return_pred=True)
        exp = explainer.explain_instance(doc_to_explain, pipeline.predict)
        print('Document id: %d' % idx)
        print(newsgroups_test.data[idx])
        print('Probability(christian) =', pipeline.predict([newsgroups_test.data[idx]])[0, 1])
        print('True class: %s' % dataset_encapsulation.class_names[newsgroups_test.target[idx]])
        print(exp.as_list())

        exp.save_to_file(os.path.join(folder,f'testpostTraining{idx}_target{newsgroups_test.target[idx]}.html'))
        pred = torch.exp(pred).cpu().detach().numpy()

        ourexplainer = TextExplainMapper(doc_to_explain, dataset_encapsulation.vectorizer, dataset_encapsulation.class_names, pred[0])
        ourexplainer.explain(sample_list[0].detach().numpy().astype(np.float64))
        ourexplainer.save_to_file(os.path.join(folder,f'testours{idx}_target{newsgroups_test.target[idx]}.html'))
