from ..EvaluationUtils import define_target, calculate_cost
from ..utils.utils import *
from ..PytorchDistributionUtils import *
import torch.nn.functional as F
import torch.nn as nn

class ordinaryPredictionTraining():
    def __init__(self, interpretable_module, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False, loss_function = None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,):
        self.interpretable_module = interpretable_module
        self.compiled = False
        self.use_cuda = False
        self.post_hoc = post_hoc
        self.post_hoc_guidance = post_hoc_guidance
        self.argmax_post_hoc = argmax_post_hoc
        
        self.loss_function = loss_function
        self.nb_sample_z_monte_carlo = nb_sample_z_monte_carlo
        self.nb_sample_z_iwae = nb_sample_z_iwae

        if self.post_hoc_guidance is not None :
            for param in self.post_hoc_guidance.parameters():
                param.requires_grad = False
        
        if self.post_hoc and self.post_hoc_guidance is None :
            raise AttributeError("You can't have post-hoc without a post hoc guidance if you are only training the classification")


    def compile(self, optim_classification, scheduler_classification = None,):
        self.optim_classification = optim_classification
        self.scheduler_classification = scheduler_classification               
        self.compiled = True
    
    def scheduler_step(self, ):
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()

    def cuda(self):                                                                                                                                                                                                                                                           
        if not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
        else :
            self.interpretable_module.cuda()
            self.use_cuda = True
            if self.post_hoc_guidance is not None :
                self.post_hoc_guidance.cuda()
       

    def _create_dic(self, loss, loss_classification = None, loss_no_selection = None, regul_classification = None, regul_selection = None):
        dic = {}
        dic["total_loss"] = loss.item()
        if loss_classification is not None :
            dic["loss_classification"] = loss_classification.item()
        if loss_no_selection is not None :
            dic["loss_no_selection"] = loss_no_selection.item()
        if regul_classification is not None :
            dic["regul_classification"] = regul_classification.item()
        if regul_selection is not None :
            dic["regul_selection"] = regul_selection.item()
        return dic


    def define_input(self, data, target, index, dataset, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,):
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        target = define_target(data,
                        index,
                        target,
                        dim_output= dataset.get_dim_output(),
                        post_hoc = self.post_hoc,
                        post_hoc_guidance = self.post_hoc_guidance,
                        argmax_post_hoc = self.argmax_post_hoc,
                        )

        data_expanded, target_expanded, index_expanded = sampling_augmentation(data,
                                                                            target = target,
                                                                            index=index,
                                                                            mc_part = nb_sample_z_monte_carlo,
                                                                            iwae_part= nb_sample_z_iwae,
                                                                            )

        return data, target, index, data_expanded, target_expanded, index_expanded, batch_size

    

    def _train_step(self, data, target, dataset, index = None, need_dic = False, ):
        self.interpretable_module.zero_grad()
        data, target, index, data_expanded, target_expanded, index_expanded, batch_size = self.define_input(data, target, index, dataset,)
        log_y_hat, regul_classification = self.interpretable_module.prediction_module(data, index= index)

        out_loss = self.loss_function.eval(input = log_y_hat, target = target,)
        total_loss = out_loss
        if regul_classification is not None :
            total_loss += regul_classification


        total_loss = torch.mean(total_loss)
        if need_dic :
            dic = self._create_dic(total_loss, loss_classification=torch.mean(out_loss), regul_classification = regul_classification)
        else :
            dic = {}
        total_loss.backward()
        self.optim_classification.step()
        return dic







class trainingWithSelection(ordinaryPredictionTraining):
    def __init__(self, interpretable_module, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False, loss_function = None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,):   
        super().__init__(interpretable_module, post_hoc, post_hoc_guidance, argmax_post_hoc, loss_function, nb_sample_z_monte_carlo, nb_sample_z_iwae,)

    def reshape(self, mask):
        return mask


    def _train_step(self, data, target, dataset, index=None, need_dic =False, ):

        assert self.compiled, "You need to compile the training module before training"

        self.interpretable_module.zero_grad()
        data, target, index = prepare_data(data, target, index, use_cuda=self.use_cuda)
        target = define_target(data,
                index,
                target,
                dim_output= dataset.get_dim_output(),
                post_hoc = self.post_hoc,
                post_hoc_guidance = self.post_hoc_guidance,
                argmax_post_hoc = self.argmax_post_hoc,
                )

        data_expanded, target_expanded, index_expanded = sampling_augmentation(data,
                                                                            target = target,
                                                                            index=index,
                                                                            mc_part = self.nb_sample_z_monte_carlo,
                                                                            iwae_part = self.nb_sample_z_iwae,
                                                                            )

        log_y_hat, regul_classification, mask_expanded, regul_sel, p_z = self.interpretable_module(data, index= index, nb_sample_z_monte_carlo = self.nb_sample_z_monte_carlo, nb_sample_z_iwae = self.nb_sample_z_iwae, )
        out_loss = calculate_cost(
                    interpretable_module=self.interpretable_module,
                    log_y_hat = log_y_hat,
                    mask_expanded = mask_expanded,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    dim_output = dataset.get_dim_output(),
                    loss_function = self.loss_function,
                    )


        total_loss = torch.mean(out_loss, axis =0) # MEAN OVER MC SAMPLE

        if regul_classification is not None :
            total_loss += regul_classification

        if regul_sel is not None :
            total_loss += regul_sel
        total_loss = torch.mean(total_loss)
        
        if need_dic :
            dic = self._create_dic(total_loss, loss_classification=torch.mean(out_loss), regul_classification = regul_classification, regul_selection = regul_sel)
        else :
            dic = {}
        total_loss.backward()
        self.optim_classification.step()
        return dic
