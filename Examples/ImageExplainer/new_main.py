
import sys
sys.path.append("C:\\Users\\hhjs\\Desktop\\FirstProject\\MissingDataTraining\\MissingDataTraining\\")
sys.path.append("/home/hhjs/MissingDataTraining/")
from missingDataTrainingModule import *
from datasets import *
from interpretation_image import *
from default_parameter import *


from torch.distributions import *
from torch.optim import *
from functools import partial
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
if __name__ == '__main__' :

    args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler = get_default()
    final_path, trainer, loader, dic_list = experiment(args_output, args_dataset, args_classification, args_destruction, args_distribution_module, args_complete_trainer, args_train, args_test, args_compiler, name_modification = True)


    # Interpretation:
    imputation_image(trainer, loader, final_path)
    interpretation_sampled(trainer, loader, final_path)
    accuracy_output(trainer, loader, final_path, batch_size = 100)
    image_f1_score(trainer, loader, final_path, nb_samples_image = 20)