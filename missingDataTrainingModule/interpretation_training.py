from missingDataTrainingModule import PytorchDistributionUtils
from .utils import define_target, continuous_NLLLoss, MSELossLastDim, NLLLossAugmented, AccuracyLoss, calculate_cost, test_selection, test_train_loss, eval_selection
from .utils.utils import *
from functools import partial, total_ordering


import numpy as np
import torch.nn.functional as F



class SELECTION_BASED_CLASSIFICATION():
    """ Abstract class to help the classification of the different module """

    def __init__(self,
                classification_module,
                selection_module,
                distribution_module,
                monte_carlo_gradient_estimator,
                baseline = None,
                reshape_mask_function = None,
                fix_classifier_parameters = False,
                fix_selector_parameters = False,
                post_hoc = False,
                post_hoc_guidance = None,
                argmax_post_hoc = False,
                **kwargs):

        self.classification_module = classification_module
        self.selection_module = selection_module
        self.distribution_module = distribution_module
        self.monte_carlo_gradient_estimator = monte_carlo_gradient_estimator(distribution = self.distribution_module)
        self.reshape_mask_function = reshape_mask_function

        self.use_cuda = False
        self.compiled = False


      
        self.fix_classifier_parameters = fix_classifier_parameters
        self.fix_selector_parameters = fix_selector_parameters
        self.post_hoc_guidance = post_hoc_guidance
        self.post_hoc = post_hoc
        self.argmax_post_hoc = argmax_post_hoc

        if self.post_hoc_guidance is not None :
            for param in self.post_hoc_guidance.parameters():
                param.requires_grad = False

        if self.post_hoc and(self.post_hoc_guidance is None) and self.fix_classifier_parameters :
            self.post_hoc_guidance = self.classification_module

        if self.post_hoc and (self.post_hoc_guidance is None) and (not self.fix_classifier_parameters):
            raise AttributeError("You can't have post-hoc without a post hoc guidance or fixing the classifier parameters")

 
    def reshape(self, z,):
        if self.reshape_mask_function is not None :
            reshaped_z = self.reshape_mask_function(z)
            return reshaped_z
        else :
            return z

    def compile(self, optim_classification, optim_selection, scheduler_classification = None, scheduler_selection = None, optim_baseline = None, scheduler_baseline = None, optim_distribution_module = None, scheduler_distribution_module = None, **kwargs):
        self.optim_classification = optim_classification
        if self.optim_classification is None :
            self.fix_classifier_parameters = True
        self.scheduler_classification = scheduler_classification
        self.optim_selection = optim_selection
        if self.optim_selection is None :
            self.fix_selector_parameters = True
        self.scheduler_selection = scheduler_selection
        self.optim_distribution_module = optim_distribution_module
        self.scheduler_distribution_module = scheduler_distribution_module

        self.compiled = True

    def zero_grad(self):
        self.classification_module.zero_grad()
        self.selection_module.zero_grad()
        self.distribution_module.zero_grad()

    def eval(self):
        self.classification_module.eval()
        self.selection_module.eval()
        self.distribution_module.eval()

    def train(self):
        self.classification_module.train()
        self.selection_module.train()  
        self.distribution_module.train()

    def cuda(self):
        if not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
        else :
            self.classification_module.cuda()
            self.selection_module.cuda()
            self.distribution_module.cuda()
            self.use_cuda = True

    def scheduler_step(self):
        assert(self.compiled)
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()
        if self.scheduler_selection is not None :
            self.scheduler_selection.step()
        if self.scheduler_distribution_module is not None :
            self.scheduler_distribution_module.step()
        self.distribution_module.update_distribution()

    def optim_step(self):
        assert(self.compiled)
        if (not self.fix_classifier_parameters) and (self.optim_classification is not None) :
            self.optim_classification.step()
        if (not self.fix_selector_parameters) and (self.optim_selection is not None) :
            self.optim_selection.step()
        if self.optim_distribution_module is not None :
            self.optim_distribution_module.step()

    def alternate_fixing_train_epoch(self, epoch, loader, nb_step_fixed_classifier = 1, nb_step_fixed_selector = 1, nb_step_all_free = 1, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, loss_function = continuous_NLLLoss(reduction = "none"), save_dic=False, verbose = True, **kwargs):

        assert np.any(np.array([nb_step_fixed_classifier, nb_step_fixed_selector, nb_step_all_free])>0)
        if self.fix_classifier_parameters and self.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(self.compiled)
        self.train()
        total_dic = {}
        print_batch_every = len(loader.dataset_train)//loader.train_loader.batch_size//10

        
        self.last_loss_function = loss_function
        self.last_nb_sample_z_monte_carlo = nb_sample_z_monte_carlo
        self.last_nb_sample_z_iwae = nb_sample_z_iwae


        total_program = nb_step_fixed_classifier + nb_step_fixed_selector + nb_step_all_free
        init_number = np.random.randint(0, total_program+1)
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            if (batch_idx + init_number) % total_program < nb_step_fixed_classifier :
                self.fix_classifier_parameters = True
                self.fix_selector_parameters = False
            elif (batch_idx + init_number) % total_program < nb_step_fixed_classifier + nb_step_fixed_selector :
                self.fix_classifier_parameters = False
                self.fix_selector_parameters = True
            else :
                self.fix_classifier_parameters = False
                self.fix_selector_parameters = False

            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, loss_function = loss_function, need_dic= (batch_idx % print_batch_every == 0))
            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
                    self.scheduler_step()    
        return total_dic

    
    def alternate_ordinary_train_epoch(self, epoch, loader, ratio_class_selection = 2, ordinaryTraining=None, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, loss_function = continuous_NLLLoss(reduction = "none"), save_dic=False, verbose = True, **kwargs):
        
        if ordinaryTraining is None :
            raise AttributeError("ratio_class_selection is not None but ordinaryTraining is None, nothing is defined for the ratio")
        else :
            if not ordinaryTraining.compiled :
                raise AttributeError("ratio_class_selection is not None but ordinaryTraining is not compiled")
        assert ratio_class_selection>0
        if self.fix_classifier_parameters and self.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(self.compiled)
        self.train()
        total_dic = {}
        print_batch_every = len(loader.dataset_train)//loader.train_loader.batch_size//10

        
        self.last_loss_function = loss_function
        self.last_nb_sample_z_monte_carlo = nb_sample_z_monte_carlo
        self.last_nb_sample_z_iwae = nb_sample_z_iwae


        if ratio_class_selection >=1 :
            ratio_step = max(np.round(ratio_class_selection), 1)
            init_number = np.random.randint(0, ratio_step+1) # Just changing which part of the dataset is used for ordinary training and the others. TODO, there might be a more interesting way to do this, inside the dataloader for instance ?
            for batch_idx, data in enumerate(loader.train_loader):
                    data, target, index = parse_batch(data)
                    if (batch_idx + init_number) % (ratio_step+1) == 0 :
                        dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, loss_function = loss_function, need_dic= (batch_idx % print_batch_every == 0))
                        if batch_idx % print_batch_every == 0 :
                            if verbose :
                                print_dic(epoch, batch_idx, dic, loader)
                            if save_dic :
                                total_dic = save_dic_helper(total_dic, dic)
                    else :
                        dic = ordinaryTraining._train_step(data, target, loader.dataset, index=index, loss_function = loss_function)
        else :
            step_only_pred = np.round(1./ratio_class_selection).astype(int)
            init_number = np.random.randint(0, step_only_pred+1)
            for batch_idx, data in enumerate(loader.train_loader):
                data, target, index = parse_batch(data)
                if (batch_idx + init_number) % (step_only_pred+1) == 0 :
                    dic = ordinaryTraining._train_step(data, target, loader.dataset, index=index, loss_function = loss_function)
                    if batch_idx % print_batch_every == 0 :
                        if verbose :
                            print_dic(epoch, batch_idx, dic, loader)
                else :
                    dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, loss_function = loss_function, need_dic= (batch_idx % print_batch_every == 0))
                    if batch_idx % print_batch_every == 0 :
                        if verbose :
                            print_dic(epoch, batch_idx, dic, loader)
                        if save_dic :
                            total_dic = save_dic_helper(total_dic, dic)
        
        self.scheduler_step()    
        return total_dic

    def classic_train_epoch(self, epoch, loader, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, save_dic=False, verbose=True, loss_function = continuous_NLLLoss(reduction = "none"),**kwargs):
        if self.fix_classifier_parameters and self.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(self.compiled)
        self.train()
        total_dic = {}
        print_batch_every = len(loader.dataset_train)//loader.train_loader.batch_size//10

        
        self.last_loss_function = loss_function
        self.last_nb_sample_z_monte_carlo = nb_sample_z_monte_carlo
        self.last_nb_sample_z_iwae = nb_sample_z_iwae



        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, loss_function = loss_function, need_dic= (batch_idx % print_batch_every == 0))
            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

        self.scheduler_step()    
        return total_dic



    def _create_dic(self, loss_total, pi_list, loss_rec = None, loss_reg = None, loss_selection = None, ):
        # dic = super()._create_dic(loss_total, neg_likelihood, mse_loss)
        dic = {}
        dic["loss_total"] = loss_total.detach().cpu().item()
        dic["mean_pi_list"] = torch.mean(torch.mean(pi_list.flatten(1),1)).item()
        quantiles = torch.tensor([0.25,0.5,0.75])
        if self.use_cuda: 
            quantiles = quantiles.cuda()
        q = torch.quantile(pi_list.flatten(1),quantiles,dim=1,keepdim = True)
        dic["pi_list_median"] = torch.mean(q[1]).item()
        dic["pi_list_q1"] = torch.mean(q[0]).item()
        dic["pi_list_q2"] = torch.mean(q[2]).item()
        if self.classification_module.imputation.has_constant():
            if torch.is_tensor(self.classification_module.imputation.get_constant()):
                dic["constantLeanarble"]= self.classification_module.imputation.get_constant().item()
        
        dic["loss_rec"] = get_item(loss_rec)
        dic["loss_reg"] = get_item(loss_reg)
        dic["loss_selection"] = get_item(loss_selection)
        return dic

    
    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 3, nb_sample_z_iwae = 3, loss_function = continuous_NLLLoss(reduction = "none"), need_dic = False):
        self.zero_grad()
        if self.monte_carlo_gradient_estimator.fix_n_mc :
            nb_sample_z_monte_carlo = 2**(np.prod(data.shape[1:])*nb_sample_z_iwae)

        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                        target = target,
                                                                                                        index=index,
                                                                                                        one_hot_target = one_hot_target,
                                                                                                        mc_part = nb_sample_z_monte_carlo,
                                                                                                        iwae_part= nb_sample_z_iwae,
                                                                                                        )
        
        # Selection Module :
        log_pi_list, loss_reg = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)

        cost_calculation = partial(calculate_cost,
                        trainer = self,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dataset.get_dim_output(),
                        loss_function = loss_function,
                        )

        loss_s, loss_f = self.monte_carlo_gradient_estimator(cost_calculation, pi_list, nb_sample_z_monte_carlo)
        
        if self.monte_carlo_gradient_estimator.combined_grad_f_s :
            loss_total = loss_reg + loss_s # How to treat differently for REINFORCE or REPARAM ?
        else :
            loss_total = loss_reg + loss_s + loss_f

        torch.mean(loss_total).backward()
        self.optim_step()

        if need_dic :
            dic = self._create_dic(loss_total = torch.mean(loss_total),
                        loss_rec = torch.mean(loss_f),
                        loss_reg = torch.mean(loss_reg),
                        loss_selection = torch.mean(loss_s),
                        pi_list = torch.exp(log_pi_list),
                        )
        else :
            dic = {}
        return dic

    
    def sample_z(self, data, target, index, dataset, nb_sample_z_monte_carlo, nb_sample_z_iwae):
        batch_size = data.shape[0]
        log_pi_list, _ = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)
        p_z = self.distribution_module(probs = pi_list)
        z = self.distribution_module.sample((nb_sample_z_monte_carlo,))
        return z

    def get_pi_list(self, loader,):
        pi_list_total = []
        dic = {}
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                if self.use_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)
                log_pi_list, _ = self.selection_module(data)
                pi_list_total.append(torch.exp(log_pi_list).cpu().numpy())
        
        treated_pi_list_total = np.concatenate(pi_list_total)
        dic["mean_pi_list"] = np.mean(treated_pi_list_total).item()
        q = np.quantile(treated_pi_list_total, [0.25,0.5,0.75])
        dic["pi_list_q1"] = q[0].item()
        dic["pi_list_median"] = q[1].item()
        dic["pi_list_q2"] = q[2].item()

        return dic
        

    def test(self, epoch, loader, liste_mc = [(1,1,1,1), (100,1,1,1), (1,100,1,1), (1,1,100,1), (1,1,1,100)]):
        """
        Do multiple test with/without sel, with different number of MC samples for mask sampling and imputation sampling.
        """
        print("\nTest epoch {}".format(epoch))
        total_dic = {}
        total_dic.update(test_train_loss(trainer = self, loader = loader, loss_function = self.last_loss_function, nb_sample_z_monte_carlo = self.last_nb_sample_z_monte_carlo, nb_sample_z_iwae = self.last_nb_sample_z_iwae, mask_sampling = self.sample_z,))
        total_dic.update(eval_selection(trainer = self, loader = loader))
        self.cuda() # QUICK FIX BECAUE SELECTION TEST THROW OUT OF CUDA @HHJS TODO LEAVE ON CUDA``
        total_dic.update(test_selection(trainer = self, loader = loader, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1,))

        for mc_config in liste_mc :
            nb_sample_z_monte_carlo = mc_config[0]
            nb_sample_z_iwae = mc_config[1]
            nb_imputation_mc = mc_config[2]
            nb_imputation_iwae = mc_config[3]
            self.classification_module.imputation.nb_imputation_mc_test = nb_imputation_mc
            self.classification_module.imputation.nb_imputation_iwae_test = nb_imputation_iwae
            total_dic.update(test_selection(trainer = self, loader = loader, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_iwae = nb_sample_z_iwae, mask_sampling = self.sample_z))
        total_dic.update(self.get_pi_list(loader = loader,))

        return total_dic



class REALX(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module,
                selection_module,
                distribution_module,
                monte_carlo_gradient_estimator,
                classification_distribution_module = PytorchDistributionUtils.wrappers.FixedBernoulli(),
                baseline = None,
                reshape_mask_function = None,
                fix_classifier_parameters = False,
                fix_selector_parameters = False,
                post_hoc = False,
                post_hoc_guidance = None,
                argmax_post_hoc = False,
                ratio_class_selection = None,
                ordinaryTraining = None,):

        super().__init__(classification_module = classification_module,
                        selection_module = selection_module,
                        distribution_module = distribution_module,
                        monte_carlo_gradient_estimator = monte_carlo_gradient_estimator,
                        baseline = baseline,
                        reshape_mask_function = reshape_mask_function,
                        fix_classifier_parameters = fix_classifier_parameters,
                        fix_selector_parameters = fix_selector_parameters,
                        post_hoc = post_hoc,
                        post_hoc_guidance = post_hoc_guidance,
                        argmax_post_hoc = argmax_post_hoc,
                        ratio_class_selection = ratio_class_selection,
                        ordinaryTraining = ordinaryTraining,
                        )

        # if self.post_hoc_guidance is not None :
            # raise NotImplementedError("REALX does not support post hoc guidance")
        self.classification_distribution_module = classification_distribution_module

    def _create_dic(self,loss_total,loss_rec_evalx,loss_rec,loss_reg,loss_selection,pi_list):
        dic = super()._create_dic(loss_total = loss_total, pi_list = pi_list, loss_rec = loss_rec, loss_reg = loss_reg, loss_selection = loss_selection)
        dic["loss_rec_evalx"] = loss_rec_evalx.detach().cpu().item()
        return dic


    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 1, nb_sample_z_iwae = 1, loss_function = continuous_NLLLoss(reduction = "none"), need_dic = False,):
        self.zero_grad()
        if self.monte_carlo_gradient_estimator.fix_n_mc :
            nb_sample_z_monte_carlo = 2**(np.prod(data.shape[1:])*nb_sample_z_iwae)
        
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        target, one_hot_target = define_target(data, index, target, one_hot_target = one_hot_target, post_hoc = self.post_hoc, post_hoc_guidance = self.post_hoc_guidance, argmax_post_hoc = self.argmax_post_hoc)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = sampling_augmentation(data,
                                                                                                target = target,
                                                                                                index=index,
                                                                                                one_hot_target = one_hot_target,
                                                                                                mc_part = nb_sample_z_monte_carlo,
                                                                                                iwae_part = nb_sample_z_iwae,
                                                                                                )


        # Destructive module :
        log_pi_list, loss_reg = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_iwae, -1) # IWae is part of the parameters while monte carlo is used in the monte carlo gradient estimator.
        pi_list = torch.exp(log_pi_list)


        #### TRAINING CLASSIFICATION :
        
        # Train classification module :
        p_z = self.classification_distribution_module(pi_list)
        z = self.classification_distribution_module.sample(sample_shape = (nb_sample_z_monte_carlo,))
        z = self.reshape(z)
        
        loss_classification = calculate_cost(mask_expanded = z,
                        trainer = self,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dataset.get_dim_output(),
                        loss_function = loss_function,
                        )

        loss_classification = loss_classification.mean(axis = 0) # Mean on MC SAmples here

        if not self.fix_classifier_parameters :
            torch.mean(loss_classification, axis=0).backward()
            self.optim_classification.step()
            self.zero_grad()


        cost_calculation = partial(calculate_cost,
                        trainer = self,
                        data_expanded = data_expanded,
                        target_expanded = target_expanded,
                        index_expanded = index_expanded,
                        one_hot_target_expanded = one_hot_target_expanded,
                        dim_output = dataset.get_dim_output(),
                        loss_function = loss_function,
                        )

        loss_s, loss_f = self.monte_carlo_gradient_estimator(cost_calculation, pi_list, nb_sample_z_monte_carlo)


        loss_total = loss_reg + loss_s #  How to treat differently for REINFORCE or REPARAM ?
        if not self.fix_selector_parameters :
            torch.mean(loss_total).backward()
            self.optim_selection.step()
            if self.optim_distribution_module is not None :
                self.optim_distribution_module.step()

        if need_dic :
            dic = self._create_dic(loss_total = torch.mean(loss_s + loss_reg + loss_classification),
                                    loss_rec_evalx = torch.mean(loss_classification),
                                    loss_rec = torch.mean(loss_f),
                                    loss_reg = torch.mean(loss_reg),
                                    loss_selection = torch.mean(loss_s),
                                    pi_list = torch.exp(log_pi_list))
        else :
            dic = {}


        return dic


