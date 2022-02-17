from missingDataTrainingModule import PytorchDistributionUtils
from .utils_missing import *
from functools import partial


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

              

        # I am not sure the following line is necessary, something is already done in optim step
        # Maybe for a matter of speed in post hoc but whatever for now
        # TODO : check of a better way for posthoc
        # if self.fix_classifier_parameters :
        #     for param in self.classification_module.parameters():
        #         param.requires_grad = False

        if self.post_hoc_guidance is not None :
            for param in self.post_hoc_guidance.parameters():
                param.requires_grad = False

 
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

    def alternate_fixing_train_epoch(self, epoch, loader, nb_step_fixed_classifier = 1, nb_step_fixed_selector = 1, nb_step_all_free = 1, nb_sample_z_monte_carlo = 3, nb_sample_z_IWAE = 3, save_dic=False, verbose = True, **kwargs):

        assert np.any(np.array([nb_step_fixed_classifier, nb_step_fixed_selector, nb_step_all_free])>0)
        if self.fix_classifier_parameters and self.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(self.compiled)
        self.train()
        total_dic = {}
        print_batch_every = len(loader.dataset_train)//loader.train_loader.batch_size//10

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

            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE = nb_sample_z_IWAE, need_dic= (batch_idx % print_batch_every == 0))
            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
                    self.scheduler_step()    
        return total_dic

    
    def alternate_ordinary_train_epoch(self, epoch, loader, ratio_class_selection = 2, ordinaryTraining=None, nb_sample_z_monte_carlo = 3, nb_sample_z_IWAE = 3, save_dic=False, verbose = True, **kwargs):
        
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

        if ratio_class_selection >=1 :
            ratio_step = max(np.round(ratio_class_selection), 1)
            init_number = np.random.randint(0, ratio_step+1) # Just changing which part of the dataset is used for ordinary training and the others. TODO, there might be a more interesting way to do this, inside the dataloader for instance ?
            for batch_idx, data in enumerate(loader.train_loader):
                    data, target, index = parse_batch(data)
                    if (batch_idx + init_number) % (ratio_step+1) == 0 :
                        dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE = nb_sample_z_IWAE, need_dic= (batch_idx % print_batch_every == 0))
                        if batch_idx % print_batch_every == 0 :
                            if verbose :
                                print_dic(epoch, batch_idx, dic, loader)
                            if save_dic :
                                total_dic = save_dic_helper(total_dic, dic)
                    else :
                        dic = ordinaryTraining._train_step(data, target, loader.dataset, index=index)
        else :
            step_only_pred = np.round(1./ratio_class_selection).astype(int)
            init_number = np.random.randint(0, step_only_pred+1)
            for batch_idx, data in enumerate(loader.train_loader):
                data, target, index = parse_batch(data)
                if (batch_idx + init_number) % (step_only_pred+1) == 0 :
                    dic = ordinaryTraining._train_step(data, target, loader.dataset, index=index)
                    if batch_idx % print_batch_every == 0 :
                        if verbose :
                            print_dic(epoch, batch_idx, dic, loader)
                else :
                    dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE = nb_sample_z_IWAE, need_dic= (batch_idx % print_batch_every == 0))
                    if batch_idx % print_batch_every == 0 :
                        if verbose :
                            print_dic(epoch, batch_idx, dic, loader)
                        if save_dic :
                            total_dic = save_dic_helper(total_dic, dic)
        
        self.scheduler_step()    
        return total_dic

    def classic_train_epoch(self, epoch, loader, nb_sample_z_monte_carlo = 3, nb_sample_z_IWAE = 3, save_dic=False, verbose=True, **kwargs):
        if self.fix_classifier_parameters and self.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(self.compiled)
        self.train()
        total_dic = {}
        print_batch_every = len(loader.dataset_train)//loader.train_loader.batch_size//10


        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE = nb_sample_z_IWAE, need_dic= (batch_idx % print_batch_every == 0))
            if batch_idx % print_batch_every == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

        self.scheduler_step()    
        return total_dic

    def _calculate_neg_likelihood(self, data, index, log_y_hat, target,):
        """
        Calculate the negative log likelihood of the classification per element of the batch, no reduction is done

        param:
            data: the data (batch_size, nb_channel,other dim...)
            index: the index of the data (batch_size, )
            log_y_hat: the log of the output of the classification (batch_size, nb_class)
            target: the target (batch_size, )


        return:
            the negative log likelihood (batch_size, )
        """
        if not self.post_hoc:
            neg_likelihood = F.nll_loss(log_y_hat, target.flatten(), reduce = False)
        else :
            if self.post_hoc_guidance is not None : # Can handle both case where classifier is fixed or not, meaning we train a surrogate classifier or the surrogate classifier is pretrained and fixed (second case is a bit stupid)
                out_y, _ = self.post_hoc_guidance(data, index = index)
            elif self.fix_classifier_parameters : # Case where we do directly posthoc on the classifier.
                out_y, _ =self.classification_module(data, index=index)
            else :
                raise AttributeError("You can't have post-hoc without a post hoc guidance or fixing the classifier parameters")
            out_y = out_y.detach()
            if self.argmax_post_hoc :
                out_y = torch.argmax(out_y, -1)
                neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce = False)
            else :
                neg_likelihood = - torch.sum(torch.exp(out_y) * log_y_hat, -1)

        return neg_likelihood


    def _create_dic(self, loss_total, neg_likelihood, mse_loss, pi_list, loss_rec = None, loss_reg = None, loss_selection = None, ):
        # dic = super()._create_dic(loss_total, neg_likelihood, mse_loss)
        dic = {}
        dic["loss_total"] = loss_total.detach().cpu().item()
        dic['neg_likelihood'] = neg_likelihood.detach().cpu().item()
        dic['mse_loss'] = mse_loss.detach().cpu().item()
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

    
    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 3, nb_sample_z_IWAE = 3, need_dic = False):
        self.zero_grad()

        nb_imputation = self.classification_module.imputation.nb_imputation
        if self.monte_carlo_gradient_estimator.fix_n_mc :
            nb_sample_z_monte_carlo = 2**(np.prod(data.shape[1:])*nb_sample_z_IWAE)
        
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo= nb_sample_z_monte_carlo,  nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = nb_imputation)

        
        # Selection Module :
        log_pi_list, loss_reg = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_IWAE, -1)
        pi_list = torch.exp(log_pi_list)

        cost_calculation = partial(self.calculate_cost,
                        data_expanded_multiple_imputation = data_expanded_multiple_imputation,
                        target_expanded_multiple_imputation = target_expanded_multiple_imputation,
                        index_expanded_multiple_imputation = index_expanded_multiple_imputation,
                        one_hot_target_expanded_multiple_imputation = one_hot_target_expanded_multiple_imputation,
                        dim_output = dataset.get_dim_output(),
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
                        neg_likelihood = torch.mean(loss_f),
                        mse_loss = torch.tensor(0.0),
                        loss_rec = torch.mean(loss_f),
                        loss_reg = torch.mean(loss_reg),
                        loss_selection = torch.mean(loss_s),
                        pi_list = torch.exp(log_pi_list),
                        )
        else :
            dic = {}
        return dic

    def calculate_cost(self, 
                    mask_expanded,
                    data_expanded_multiple_imputation, # Shape is (nb_imputation, nb_sample_z_monte_carlo, nb_sample_z_IWAE, batch_size, channel, dim...)
                    target_expanded_multiple_imputation,
                    one_hot_target_expanded_multiple_imputation,
                    dim_output,
                    index_expanded_multiple_imputation = None,
                    ):


        nb_imputation = self.classification_module.imputation.nb_imputation
        nb_sample_z_IWAE = data_expanded_multiple_imputation.shape[3]
        mask_expanded = self.reshape(mask_expanded)

        if index_expanded_multiple_imputation is not None :
            index_expanded = index_expanded_multiple_imputation[0]
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,3)
        else :
            index_expanded_multiple_imputation_flatten = None
            index_expanded = None
        
        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,3)


        log_y_hat, _ = self.classification_module(data_expanded_multiple_imputation[0], mask_expanded, index = index_expanded)
        log_y_hat = log_y_hat.reshape(data_expanded_multiple_imputation.shape[:4] + torch.Size((dim_output,)))

        neg_likelihood = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                            index = index_expanded_multiple_imputation_flatten,
                                                            log_y_hat=log_y_hat.flatten(0,3),
                                                            target = target_expanded_multiple_imputation_flatten,
                                                            )

        neg_likelihood = neg_likelihood.reshape(data_expanded_multiple_imputation.shape[:4])
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32))
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=-1) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32))

        return neg_likelihood
  

    def test(self, loader, nb_sample_z = 10):
        mse_loss_from_mean = 0
        mse_loss = 0
        neg_likelihood_from_mean= 0
        neg_likelihood = 0
        correct_classic = 0
        correct_post_hoc = 0
        correct_destructed = 0
        correct_baseline = 0
        self.eval()
        pi_list_total = []
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                batch_size = data.shape[0]
                if self.use_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)
                one_hot_target = get_one_hot(target, num_classes = loader.dataset.get_dim_output())
                data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z, nb_imputation = None)
                if index_expanded is not None :
                    index_expanded_flatten = index_expanded.flatten(0,1)
                else :
                    index_expanded_flatten = None



                log_pi_list, _ = self.selection_module(data)
                pi_list_total.append(torch.exp(log_pi_list).cpu().numpy())
                self.distribution_module(torch.exp(log_pi_list))
                z = self.distribution_module.sample((nb_sample_z,))
                z = self.reshape(z)
                

                if self.post_hoc_guidance is not None:
                    ## Check the prediction without selection on the post-hoc-guidance method
                    log_y_hat, _ = self.post_hoc_guidance(data, index = index)
                    pred_post_hoc = torch.argmax(log_y_hat,dim = 1)
                    correct_post_hoc += pred_post_hoc.eq(target).sum().item()
 
                ## Check the prediction without selection on the baseline method
                log_y_hat, _ = self.classification_module(data, index = index)
                pred_classic = torch.argmax(log_y_hat,dim = 1)
                correct_classic += pred_classic.eq(target).sum()

                ## Check the prediction with the selection method
                log_y_hat_destructed, _ = self.classification_module(data_expanded.flatten(0,1), z, index = index_expanded_flatten)
                
                log_y_hat_destructed = log_y_hat_destructed.reshape(nb_sample_z, batch_size, loader.dataset.get_dim_output())
                log_y_hat_mean = torch.mean(log_y_hat_destructed, axis=0)
                index = torch.where(torch.any(torch.isnan(log_y_hat_destructed), axis=-1))[1]


                neg_likelihood += F.nll_loss(log_y_hat_destructed.flatten(0,1), target_expanded.flatten(0,1), reduce=False).reshape((nb_sample_z, batch_size)).mean(0).sum()
                mse_loss += F.mse_loss(torch.exp(log_y_hat_destructed.flatten(0,1)), one_hot_target_expanded.flatten(0,1), reduce=False).sum(-1).reshape((nb_sample_z, batch_size)).mean(0).sum()

                mse_loss_from_mean += torch.sum(torch.sum((torch.exp(log_y_hat_mean)-one_hot_target)**2,1))
                neg_likelihood_from_mean += F.nll_loss(log_y_hat_mean, target, reduce='sum')

                pred_destructed = torch.argmax(log_y_hat_mean, dim=1)
                correct_destructed += pred_destructed.eq(target).sum()



            mse_loss /= len(loader.test_loader.dataset) 
            neg_likelihood /= len(loader.test_loader.dataset) 
            print('\nTest set: MSE: {:.4f}, Likelihood {:.4f}, Accuracy No selection: {}/{} ({:.0f}%), Accuracy selection: {}/{} ({:.0f}%), Accuracy PostHoc: {}/{} ({:.0f}%),'.format(
                 mse_loss.item(), -neg_likelihood.item(),
                 correct_classic.item(), len(loader.test_loader.dataset),100. * correct_classic.item() / len(loader.test_loader.dataset),
                 correct_destructed.item(), len(loader.test_loader.dataset), 100. * correct_destructed.item() / len(loader.test_loader.dataset),
                 correct_post_hoc, len(loader.test_loader.dataset), 100. * correct_post_hoc / len(loader.test_loader.dataset),
                ))
            print("\n")
            total_dic = self._create_dic_test(correct_destructed/len(loader.test_loader.dataset),
                correct_classic/len(loader.test_loader.dataset),
                neg_likelihood,
                mse_loss,
                neg_likelihood_from_mean,
                mse_loss_from_mean,
                pi_list_total,
                correct_post_hoc=correct_post_hoc)

        return total_dic

    def _create_dic_test(self, correct, correct_no_selection, neg_likelihood, mse_loss, neg_likelihood_from_mean, mse_loss_from_mean, pi_list_total, correct_post_hoc = None):
        total_dic = {}
        total_dic["accuracy_prediction_no_selection"] = correct_no_selection.item()
        total_dic["neg_likelihood"] = neg_likelihood.item()
        total_dic["mse_loss"] = mse_loss.item()
        total_dic["neg_likelihood_from_mean"] = neg_likelihood_from_mean.item()
        total_dic["mse_loss_from_mean"] = mse_loss_from_mean.item()
        treated_pi_list_total = np.concatenate(pi_list_total)
        total_dic["mean_pi_list"] = np.mean(treated_pi_list_total).item()
        total_dic["accuracy_prediction_selection"] = correct.item()
        q = np.quantile(treated_pi_list_total, [0.25,0.5,0.75])
        total_dic["pi_list_q1"] = q[0].item()
        total_dic["pi_list_median"] = q[1].item()
        total_dic["pi_list_q2"] = q[2].item()
        total_dic["correct_post_hoc"] = correct_post_hoc

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

        if self.post_hoc_guidance is not None :
            raise NotImplementedError("REALX does not support post hoc guidance")
        self.classification_distribution_module = classification_distribution_module


    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 1, nb_sample_z_IWAE = 1, need_dic = False):
        self.zero_grad()
        nb_imputation = self.classification_module.imputation.nb_imputation
        if self.monte_carlo_gradient_estimator.fix_n_mc :
            nb_sample_z_monte_carlo = 2**(np.prod(data.shape[1:])*nb_sample_z_IWAE)
        
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo= nb_sample_z_monte_carlo,  nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = nb_imputation)

        # Destructive module :
        log_pi_list, loss_reg = self.selection_module(data)
        log_pi_list = log_pi_list.unsqueeze(1).expand(batch_size, nb_sample_z_IWAE, -1)
        pi_list = torch.exp(log_pi_list)


        #### TRAINING CLASSIFICATION :
        
        # Train classification module :
        p_z = self.classification_distribution_module(pi_list)
        z = self.classification_distribution_module.sample(sample_shape = (nb_sample_z_monte_carlo,))
        z = self.reshape(z)
        


        # Selection Module :
        loss_classification = self.calculate_cost(mask_expanded = z,
                        data_expanded_multiple_imputation = data_expanded_multiple_imputation,
                        target_expanded_multiple_imputation = target_expanded_multiple_imputation,
                        index_expanded_multiple_imputation = index_expanded_multiple_imputation,
                        one_hot_target_expanded_multiple_imputation = one_hot_target_expanded_multiple_imputation,
                        dim_output = dataset.get_dim_output(),
                        )

        loss_classification = loss_classification.mean(axis = 0)

        if not self.fix_classifier_parameters :
            torch.mean(loss_classification, axis=0).backward()
            self.optim_classification.step()
            self.zero_grad()


        cost_calculation = partial(self.calculate_cost,
                        data_expanded_multiple_imputation = data_expanded_multiple_imputation,
                        target_expanded_multiple_imputation = target_expanded_multiple_imputation,
                        index_expanded_multiple_imputation = index_expanded_multiple_imputation,
                        one_hot_target_expanded_multiple_imputation = one_hot_target_expanded_multiple_imputation,
                        dim_output = dataset.get_dim_output(),
                        )

        loss_s, neg_likelihood = self.monte_carlo_gradient_estimator(cost_calculation, pi_list, nb_sample_z_monte_carlo)


        loss_total = loss_reg + loss_s #  How to treat differently for REINFORCE or REPARAM ?
        if not self.fix_selector_parameters :
            torch.mean(loss_total).backward()
            self.optim_selection.step()
            if self.optim_distribution_module is not None :
                self.optim_distribution_module.step()

        if need_dic :
            dic = self._create_dic(loss_total = torch.mean(loss_s + loss_reg + loss_classification),
                                    neg_likelihood = torch.mean(neg_likelihood),
                                    mse_loss = torch.tensor(0.0).type(torch.float32),
                                    loss_rec = torch.mean(loss_classification),
                                    loss_reg = torch.mean(loss_reg),
                                    loss_selection = torch.mean(loss_s),
                                    pi_list = torch.exp(log_pi_list))
        else :
            dic = {}


        return dic


