import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn

from ..utils.utils import *
from .classification_evaluation import *
from .selection_evaluation import *

    
    
def test_epoch(interpretable_module, epoch, loader, args, liste_mc = [(1,1,1,1),], trainer = None,):
    """
    Do multiple test with/without sel, with different number of MC samples for mask sampling and imputation sampling.
    """
    print("==========================================================")
    print("\nTest epoch {} started".format(epoch))
    total_dic = {}
    if epoch is not None :
        total_dic["epoch"] = epoch

    total_dic.update(multiple_test(interpretable_module= interpretable_module, loader = loader,))

    if hasattr(interpretable_module, "sample_z"):
        sample_z_function = interpretable_module.sample_z
    else :
        sample_z_function = None

    original_nb_imputation_mc_train = interpretable_module.prediction_module.imputation.nb_imputation_mc
    original_nb_imputation_iwae_train = interpretable_module.prediction_module.imputation.nb_imputation_iwae
    original_nb_imputation_mc_test = interpretable_module.prediction_module.imputation.nb_imputation_mc_test
    original_nb_imputation_iwae_test = interpretable_module.prediction_module.imputation.nb_imputation_iwae_test


    if trainer is not None :
        interpretable_module.prediction_module.imputation.nb_imputation_mc_test = original_nb_imputation_mc_train
        interpretable_module.prediction_module.imputation.nb_imputation_iwae_test = original_nb_imputation_iwae_train
        total_dic.update(
            test_train_loss(interpretable_module = interpretable_module,
                loader = loader,
                loss_function = trainer.loss_function,
                nb_sample_z_monte_carlo = trainer.nb_sample_z_monte_carlo,
                nb_sample_z_iwae = trainer.nb_sample_z_iwae,
                mask_sampling = sample_z_function,
                trainer = trainer,
                ),
            )
        if hasattr(interpretable_module, "sample_z"):
            total_dic.update(multiple_test(interpretable_module = interpretable_module,
                        loader = loader,
                        nb_sample_z_monte_carlo = trainer.nb_sample_z_monte_carlo,
                        nb_sample_z_iwae = trainer.nb_sample_z_iwae,
                        trainer = trainer,
                        mask_sampling = sample_z_function,))
        
    
    
    if hasattr(interpretable_module, "sample_z"):
        if hasattr(loader.dataset, "optimal_S_test") :
            total_dic.update(eval_selection(interpretable_module = interpretable_module, loader = loader, args = args,))
        
        if args.args_train.use_cuda :
            interpretable_module.cuda() # QUICK FIX BECAUSE SELECTION TEST THROW OUT OF CUDA @HHJS TODO LEAVE ON CUDA``

        

        for mc_config in liste_mc :
            nb_sample_z_monte_carlo = mc_config[0]
            nb_sample_z_iwae = mc_config[1]
            nb_imputation_mc = mc_config[2]
            nb_imputation_iwae = mc_config[3]
            interpretable_module.prediction_module.imputation.nb_imputation_mc_test = nb_imputation_mc
            interpretable_module.prediction_module.imputation.nb_imputation_iwae_test = nb_imputation_iwae
            total_dic.update(
                multiple_test(interpretable_module = interpretable_module,
                            loader = loader,
                            nb_sample_z_monte_carlo = nb_sample_z_monte_carlo,
                            nb_sample_z_iwae = nb_sample_z_iwae,
                            mask_sampling = interpretable_module.sample_z,
                            trainer = trainer,
                ),
            )

        interpretable_module.prediction_module.imputation.nb_imputation_mc_test = original_nb_imputation_mc_test
        interpretable_module.prediction_module.imputation.nb_imputation_iwae_test = original_nb_imputation_iwae_test
    
    if hasattr(interpretable_module, "EVALX"):
        total_dic.update(multiple_test(interpretable_module = interpretable_module.EVALX,
                            loader = loader,
                            nb_sample_z_monte_carlo = trainer.nb_sample_z_monte_carlo,
                            nb_sample_z_iwae = trainer.nb_sample_z_iwae,
                            mask_sampling = interpretable_module.EVALX.sample_z,
                            trainer = trainer,
                            prefix = "EVALX_",
                            ))

    print("\nTest epoch {} done".format(epoch))
    print("==========================================================")


    return total_dic