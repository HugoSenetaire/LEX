import os
import sys
current_file_path = os.path.abspath(__file__)
while(not current_file_path.endswith("MissingDataTraining")):
    current_file_path = os.path.dirname(current_file_path)
sys.path.append(current_file_path)

from missingDataTrainingModule import *
from torch.distributions import *
from torch.optim import *
from datasets import *
from interpretation_image import *




def get_dataset(args_dataset):
    dic_args_dataset = vars(args_dataset)
    dataset = args_dataset.dataset(**dic_args_dataset)
    loader = args_dataset.loader(dataset, batch_size_train=args_dataset.batch_size_train, batch_size_test=args_dataset.batch_size_test,)
    return dataset, loader


def multiple_experiment(count,
                        dataset,
                        loader,
                        args,
                        name_modification = False,
                        nb_samples_image_per_category = 20,
                        batch_size_test = 100,):
    try :
        final_path, trainer, loader, dic_list = main_launcher.experiment(dataset,
                                                            loader,
                                                            complete_args=args,
                                                            )

        complete_analysis_image(trainer, loader, final_path, batch_size = batch_size_test, nb_samples_image_per_category = nb_samples_image_per_category)
        return count+1
    except Exception as e:
        print(e)
        if os.path.exists(args.args_output.path):
            os.rename(args.args_output.path, args.args_output.path+"_error")
        else :
            if not os.path.exists(args.args_output.path+"_error"):
                os.makedirs(args.args_output.path+"_error")
        with open(args.args_output.path+"_error/error.txt", "w") as f:
            f.write(str(e))
        return count+1