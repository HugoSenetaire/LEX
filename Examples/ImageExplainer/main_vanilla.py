import sys
sys.path.append("D:\\DTU\\firstProject\\MissingDataTraining")
from missingDataTrainingModule import *
from datasets import *



from torch.distributions import *
from torch.optim import *

if __name__ == "__main__":

    input_size_classifier = (1, 28, 28)
    dataset_class = FashionMNISTDataset
    mnist = LoaderEncapsulation(dataset_class,64,1000)

    classifier = StupidClassifier(input_size_classifier, mnist.get_category())
    classification_vanilla = ClassificationModule(classifier)
    optim_classifier = Adam(classification_vanilla.parameters())
    trainer_vanilla = ordinaryTraining(classification_vanilla)
    feature_extractor= None


    folder = "D:\DTU\Interpretable_or_not"
    folder = os.path.join(folder,dataset_class.__name__)
    if not os.path.exists(folder):
        os.makedirs(folder)
    experiment_name = 'Vanilla'
    final_path = os.path.join(folder,experiment_name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)


    data, target= next(iter(mnist.test_loader))
    data = data[:10]
    target = target[:10]
    print(data)
    print(target)
    for epoch in range(10):
        trainer_vanilla.train(epoch, mnist,optim_classifier)
        trainer_vanilla.test(mnist)


    
    pipeline_predict = partial(batch_predict_gray, model = classifier, feature_extractor= feature_extractor)
    lime_image_interpretation(final_path, "lime_explained", data, target, pipeline_predict)


    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=20, ratio=0.2)
    lime_image_interpretation(final_path, "more_granularity", data, target, pipeline_predict,segmenter=segmenter)

    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=5, ratio=0.2)
    lime_image_interpretation(final_path, "even_more_granularity", data, target, pipeline_predict,segmenter=segmenter)



