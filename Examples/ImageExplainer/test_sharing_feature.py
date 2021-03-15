import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *



from torch.distributions import *
from torch.optim import *

from functools import partial

from lime import lime_image
from skimage.segmentation import mark_boundaries

if __name__ == "__main__":
    dataset = MnistVariationFashion

    
    input_size_classifier = (1,28,28)
    input_size_destructor = (1,28,28)
    stride_patch = (1,1)
    kernel_patch = (1,1)

    lambda_reg = 0.1

    lr = 1e-4
    nb_epoch=10
    epoch_pretrain = 0
    training_vanilla = 1

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

    path_save = r"D:\DTU\SharingFeature"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    folder = os.path.join(path_save,dataset.__name__)
    if not os.path.exists(folder):
        os.makedirs(folder)

    experiment_name = "SharingSimple"

    final_path = os.path.join(folder, experiment_name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    input_size_destructor = (1,28,28)
    input_size_autoencoder = (1,28,28)
    input_size_classifier = (1,28,28)
    input_size_classification_module = (1, 28, 28)

 

    mnist = LoaderEncapsulation(dataset)
    data, target= next(iter(mnist.test_loader))
    data = data[:10]
    target = target[:10]

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


    #============ No Variationnal ===============
    feature_extractor = FeatureExtraction()


   
    data, target= next(iter(mnist.test_loader))
    data = data[:10]
    target = target[:10]

          
    parameter = {
        "train_reconstruction": train_reconstruction,
        "train_postprocess" : train_postprocess,
        "recons_reg" : reconstruction_regularization,
        "post_process_reg" : post_process_regularization,
        "epoch_pretrain": epoch_pretrain,
        "noise_function":noise_function,
        "lambda_reg":lambda_reg,
        "post_process_regularization":post_process_regularization,
        "reconstruction_regularization":reconstruction_regularization,
        "feature_extractor": feature_extractor
    }

    if need_autoencoder :
        save_interpretation(final_path,
        data_autoencoder.detach().cpu().numpy(), 
        target_autoencoder.detach().cpu().numpy(), [0,1,2,3],prefix= "input_autoencoder")

        autoencoder_network_missing = copy.deepcopy(autoencoder_network)
        output = autoencoder_network_missing(data_autoencoder.cuda()).reshape(data_autoencoder.shape)

        save_interpretation(final_path,
        output.detach().cpu().numpy(), 
        target_autoencoder.detach().cpu().numpy(), [0,1,2,3], prefix="output_autoencoder_before_training")

    ##### ============ Classification training ========= :
    if feature_extractor is not None :
        classifier_no_var = ClassifierFromFeature()
    else :
        classifier_no_var = ClassifierModel()
    imputation = ConstantImputation(input_size= input_size_classification_module,post_process_regularization = post_process_regularization,
                                                 reconstruction_reg= reconstruction_regularization)
    classification_no_var = ClassificationModule(classifier_no_var, imputation=imputation, feature_extractor=feature_extractor)


    if feature_extractor is not None :
        destructor_no_var = DestructorFromFeature()
    else :
        destructor_no_var = DestructorSimilar()

    destruction_no_var = DestructionModule(destructor_no_var, regularization=free_regularization,  feature_extractor=feature_extractor)

    trainer_no_var = noVariationalTraining(classification_no_var, destruction_no_var, feature_extractor=feature_extractor, kernel_patch = kernel_patch, stride_patch = stride_patch)

    
    temperature = 1.0
    optim_classification = Adam(classification_no_var.parameters())
    optim_destruction = Adam(destruction_no_var.parameters())
    if feature_extractor is not None :
        optim_feature_extractor = Adam(feature_extractor.parameters())
    else :
        optim_feature_extractor = None



    total_dic_train = {}
    total_dic_test_no_var = {}

    for epoch in range(nb_epoch):
        
        dic_train = trainer_no_var.train(
            epoch, mnist,
            optim_classification, optim_destruction,
            partial(RelaxedBernoulli,temperature),
            optim_feature_extractor = optim_feature_extractor,
            lambda_reg=lambda_reg,
            lambda_reconstruction = 0.1,
            save_dic = True
        )
        dic_test_no_var = trainer_no_var.test_no_var(mnist, Bernoulli)
        temperature *= 0.5

        total_dic_train = fill_dic(total_dic_train, dic_train)
        total_dic_test_no_var = fill_dic(total_dic_test_no_var, dic_test_no_var)



    ###### Sample and save results
    save_dic(os.path.join(final_path,"train"), total_dic_train)
    save_dic(os.path.join(final_path,"test_no_var"), total_dic_test_no_var)


    sample_list, pred = trainer_no_var.MCMC(mnist,data, target, Bernoulli,5000, return_pred=True)
    save_interpretation(final_path,sample_list, data, target, suffix = "no_var",
                        y_hat = torch.exp(pred).detach().cpu().numpy(),
                        class_names=[str(i) for i in range(10)])


    pipeline_predict = partial(batch_predict_gray, model = classifier_no_var, feature_extractor = feature_extractor)
    lime_path = os.path.join(final_path, "lime_explained")
    if not os.path.exists(lime_path):
        os.makedirs(lime_path)
    lime_image_interpretation(lime_path, data, target, pipeline_predict)

    lime_path_granularity = os.path.join(final_path, "lime_explained_more_granularity")
    if not os.path.exists(lime_path_granularity):
        os.makedirs(lime_path_granularity)

    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=20, ratio=0.2)
    lime_image_interpretation(lime_path_granularity, data, target, pipeline_predict,segmenter=segmenter)



    lime_path_destruction = os.path.join(final_path, "lime_explained_destruction")
    if not os.path.exists(lime_path_destruction):
        os.makedirs(lime_path_destruction)
    prediction_function = partial(trainer_no_var._predict, sampling_distribution = partial(RelaxedBernoulli,temperature), dataset = mnist)
    pipeline_predict = partial(batch_predict_gray_with_destruction, model_function=prediction_function)
    lime_image_interpretation(lime_path_destruction, data, target, pipeline_predict,segmenter=segmenter)


    