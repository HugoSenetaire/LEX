from .utils import define_target, continuous_NLLLoss, MSELossLastDim, NLLLossAugmented, AccuracyLoss, calculate_cost, multiple_test, test_train_loss
from .utils.utils import *
from .PytorchDistributionUtils import *
import torch.nn.functional as F
import torch.nn as nn




class ordinaryTraining():
    def __init__(self, classification_module, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False,):    
        self.classification_module = classification_module
        self.compiled = False
        self.use_cuda = False
        self.post_hoc = post_hoc
        self.post_hoc_guidance = post_hoc_guidance
        self.argmax_post_hoc = argmax_post_hoc
        

        if self.post_hoc_guidance is not None :
            for param in self.post_hoc_guidance.parameters():
                param.requires_grad = False
        
        if self.post_hoc and self.post_hoc_guidance is None :
            raise AttributeError("You can't have post-hoc without a post hoc guidance if you are only training the classification")


    def compile(self, optim_classification, scheduler_classification = None,):
        self.optim_classification = optim_classification
        self.scheduler_classification = scheduler_classification
        self.compiled = True

    def cuda(self):
        if not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
        else :
            self.classification_module.cuda()
            self.use_cuda = True
       

    def _create_dic(self, loss, loss_no_selection = None, ):
        dic = {}
        dic["total_loss"] = loss.item()
        if loss_no_selection is not None :
            dic["loss_no_selection"] = loss_no_selection.item()
        return dic


    def zero_grad(self):
        self.classification_module.zero_grad()

    def train(self):
        self.classification_module.train()

    def eval(self):
        self.classification_module.eval()

    def _train_step(self, data, target, dataset, index = None, loss_function = NLLLossAugmented(reduction='none'),):
        self.zero_grad()
        data, target, one_hot_target, index = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        target, one_hot_target = define_target(data, index, target, one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc, dim_output= dataset.get_dim_output(),)
        batch_size = data.shape[0]
        log_y_hat, _ = self.classification_module(data, index= index)

        loss_rec = loss_function.eval(input = log_y_hat, target = target, one_hot_target = one_hot_target)
        loss = torch.mean(loss_rec)
        dic = self._create_dic(loss, )
        loss.backward()
        self.optim_classification.step()
        return dic


    def train_epoch(self, epoch, loader, loss_function=NLLLossAugmented(reduction='none'),save_dic = False, verbose = False,):
        self.train()
        print_batch_every =  max(len(loader.dataset_train)//loader.train_loader.batch_size//10, 1)
        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)

            dic = self._train_step(data, target, loader.dataset, index=index, loss_function= loss_function,)

            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()
        
        return total_dic


    def test(self, epoch, loader, ):
        """
        Do multiple test with/without sel, with different number of MC samples for mask sampling and imputation sampling.
        """
        print("\nTest epoch {}".format(epoch))
        total_dic = {}
        total_dic["epoch"] = epoch
        total_dic.update(multiple_test(trainer = self, loader = loader, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,))
        return total_dic



class trueSelectionTraining(ordinaryTraining):
    def __init__(self, classification_module, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False,):   
        super().__init__(classification_module, post_hoc, post_hoc_guidance, argmax_post_hoc)

    def reshape(self, mask):
        return mask
    
    def _train_step(self, data, target, dataset, index=None, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, loss_function = continuous_NLLLoss(reduction = "none") ):
        self.zero_grad()
        data, target, one_hot_target, index = prepare_data(data, target, index, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc, dim_output= dataset.get_dim_output(),)

        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                        target = target,
                                                                                                        index=index,
                                                                                                        one_hot_target = one_hot_target,
                                                                                                        mc_part = nb_sample_z_monte_carlo,
                                                                                                        iwae_part = nb_sample_z_iwae,
                                                                                                        )

       

        true_mask = dataset.optimal_S_train[index].type(torch.float32).to(data.device)
        true_mask = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)

        out_loss = calculate_cost(
                    trainer = self,
                    mask_expanded = true_mask,
                    data_expanded = data_expanded,
                    target_expanded = target_expanded,
                    index_expanded = index_expanded,
                    one_hot_target_expanded = one_hot_target_expanded,
                    dim_output = dataset.get_dim_output(),
                    loss_function = loss_function,
                    )
        loss = torch.mean(out_loss, dim=0)
        loss = torch.mean(loss)
        dic = self._create_dic(loss, loss,)
        loss.backward()
        self.optim_classification.step()
        return dic

    def train_epoch(self, epoch, loader, loss_function = continuous_NLLLoss(reduction='none'), save_dic = False, verbose = False,):
        self.train()
        print_batch_every =  max(len(loader.dataset_train)//loader.train_loader.batch_size//10, 1)
        
        self.last_loss_function = loss_function
        self.last_nb_sample_z_monte_carlo = 1
        self.last_nb_sample_z_iwae = 1

        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            dic = self._train_step(data, target, dataset = loader.dataset, index=index, loss_function=loss_function,)
            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()
        
        return total_dic

    def sample_z(self, data, target, index, dataset, nb_sample_z_monte_carlo, nb_sample_z_iwae):
        true_mask = dataset.optimal_S_train[index].type(torch.float32).to(data.device)
        true_mask = extend_input(true_mask, mc_part = nb_sample_z_monte_carlo, iwae_part = nb_sample_z_iwae)
        return true_mask
        

    def test(self, epoch, loader, liste_mc = [(1,1,1,1), (100,1,1,1), (1,100,1,1), (1,1,100,1), (1,1,1,100)]):
        total_dic = super().test(epoch, loader)
        total_dic.update(test_train_loss(trainer = self, loader = loader, loss_function = self.last_loss_function, nb_sample_z_monte_carlo = self.last_nb_sample_z_monte_carlo, nb_sample_z_iwae = self.last_nb_sample_z_iwae, mask_sampling = self.sample_z,))

        for mc_config in liste_mc :
            nb_sample_z_monte_carlo = mc_config[0]
            nb_sample_z_iwae = mc_config[1]
            nb_imputation_mc = mc_config[2]
            nb_imputation_iwae = mc_config[3]
            self.classification_module.imputation.nb_imputation_mc_test = nb_imputation_mc
            self.classification_module.imputation.nb_imputation_iwae_test = nb_imputation_iwae
            total_dic.update(multiple_test(trainer = self, loader = loader, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, mask_sampling = self.sample_z))
        return total_dic


class EVAL_X(ordinaryTraining):
    def __init__(self, classification_module, fixed_distribution = wrappers.FixedBernoulli(),
                reshape_mask_function = None, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False,):
        super().__init__(classification_module, post_hoc_guidance = post_hoc_guidance, post_hoc = post_hoc, argmax_post_hoc = argmax_post_hoc)
        self.fixed_distribution = fixed_distribution
        self.reshape_mask_function = reshape_mask_function


    def train_epoch(self, epoch, loader, nb_sample_z_monte_carlo = 10, nb_sample_z_iwae = 1, loss_function = continuous_NLLLoss(reduction='none'), save_dic=False, verbose=False,):
        self.train()
        total_dic = {}
        print_batch_every =  max(len(loader.dataset_train)//loader.train_loader.batch_size//10, 1)

        self.last_loss_function = loss_function
        self.last_nb_sample_z_monte_carlo = nb_sample_z_monte_carlo
        self.last_nb_sample_z_iwae = nb_sample_z_iwae

        for batch_idx, data in enumerate(loader.train_loader):
            input, target, index = parse_batch(data)

            dic = self._train_step(input, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae=nb_sample_z_iwae, loss_function=loss_function, need_dic= (batch_idx % print_batch_every == 0))
            
            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            print(f"Learning Rate classification : {self.scheduler_classification.get_last_lr()}")
            self.scheduler_classification.step()
        
        return total_dic


    def reshape(self, z):
        if self.reshape_mask_function is not None:
            return self.reshape_mask_function(z)
        else :
            return z

    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 10, nb_sample_z_iwae = 1, loss_function = continuous_NLLLoss(reduction="none"), need_dic = False,):   
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc, dim_output= dataset.get_dim_output(),)

        nb_sample_z_monte_carlo_classification, nb_sample_z_iwae_classification = nb_sample_z_monte_carlo*nb_sample_z_iwae, 1
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                        target = target,
                                                                                                        index=index,
                                                                                                        one_hot_target = one_hot_target,
                                                                                                        mc_part = nb_sample_z_monte_carlo_classification,
                                                                                                        iwae_part= nb_sample_z_iwae_classification,
                                                                                                        )

        
        # Destructive module

        p_z = self.fixed_distribution(torch.zeros(batch_size, 1, nb_sample_z_iwae, *self.classification_module.classifier.input_size[1:]))
        # Train classification module :
        z = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo_classification,))
        # Classification module :
        loss_classification = calculate_cost(
                        trainer = self,
                        mask_expanded = z,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dataset.get_dim_output(),
                        loss_function = loss_function,
                        )



        loss_classification = loss_classification.mean(axis = 0) # Monte Carlo average
        torch.mean(loss_classification, axis=0).backward() # Batch average
        self.optim_classification.step()

        if need_dic :
            dic = self._create_dic(loss = torch.mean(loss_classification),
                        )
        else :
            dic = {}
        return dic


    def sample_z(self, data, target, index, dataset, nb_sample_z_monte_carlo, nb_sample_z_iwae):
        # Destructive module :
        data_expanded = extend_input(data, mc_part=nb_sample_z_monte_carlo, iwae_part=nb_sample_z_iwae)
        batch_size = data.shape[0]
        p_z = self.fixed_distribution(torch.zeros(batch_size, nb_sample_z_iwae, 1, *self.classification_module.classifier.input_size[1:]))
        # Train classification module :
        z = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo,))
        return z
    
    def test(self, epoch, loader, liste_mc = [(1,1,1,1), (100,1,1,1), (1,100,1,1), (1,1,100,1), (1,1,1,100)]):
        total_dic = super().test(epoch, loader)
        total_dic.update(test_train_loss(trainer = self, loader = loader, loss_function = self.last_loss_function, nb_sample_z_monte_carlo = self.last_nb_sample_z_monte_carlo, nb_sample_z_iwae = self.last_nb_sample_z_iwae, mask_sampling = self.sample_z,))
        for mc_config in liste_mc :
            nb_sample_z_monte_carlo = mc_config[0]
            nb_sample_z_iwae = mc_config[1]
            nb_imputation_mc = mc_config[2]
            nb_imputation_iwae = mc_config[3]
            self.classification_module.imputation.nb_imputation_mc_test = nb_imputation_mc
            self.classification_module.imputation.nb_imputation_iwae_test = nb_imputation_iwae
            total_dic.update(multiple_test(trainer = self, loader = loader, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, mask_sampling = self.sample_z))
        return total_dic