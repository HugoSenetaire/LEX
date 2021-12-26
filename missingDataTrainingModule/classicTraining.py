from numpy.core.fromnumeric import reshape, var
from numpy.lib.npyio import _savez_compressed_dispatcher
from numpy.testing._private.utils import requires_memory
from psutil import test
from torch import neg, neg_
from torch._C import _show_config
from torch.nn.functional import batch_norm, one_hot
from .Destruction import * 
from .Classification import *
from .Distribution import *
from .utils_missing import *


# torch.autograd.set_detect_anomaly(True)

import numpy as np
import matplotlib.pyplot as plt


complete_list = None
global_generator = None
grad ={}
grad_previous = {}
save_pz = None
global_neg_likelihood = None



class ordinaryTraining():
    def __init__(self, classification_module,):    
        self.classification_module = classification_module
        self.compiled = False
        self.use_cuda = False


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
       

    def _create_dic(self,loss, neg_likelihood, mse_loss):
        dic = {}
        dic["likelihood"] = -neg_likelihood.item()
        dic["mse_loss"] = mse_loss.item()
        dic["total_loss"] = loss.item()
        return dic

    def _create_dic_test(self, correct, neg_likelihood, mse_loss):
        dic = {}
        dic["accuracy_prediction_no_destruction"] = correct.item()
        dic["likelihood"] = -neg_likelihood.item()
        dic["mse"] = mse_loss.item()
        return dic


    def parameters(self):
        return self.classification_module.parameters()

    def zero_grad(self):
        self.classification_module.zero_grad()

    def train(self):
        self.classification_module.train()


    def _train_step(self, data, target, dataset, index = None):
        self.zero_grad()
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        log_y_hat, _ = self.classification_module(data, index= index)

        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        loss = neg_likelihood
        dic = self._create_dic(loss, neg_likelihood, mse_loss)
        loss.backward()
        self.optim_classification.step()
        
        return dic


    def train_epoch(self, epoch, loader,  save_dic = False, verbose = False,):
        self.train()

        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)

            dic = self._train_step(data, target, loader.dataset, index=index)

            if batch_idx % 100 == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            print(f"Learning Rate classification : {self.scheduler_classification.get_last_lr()}")
            self.scheduler_classification.step()
        
        return total_dic



    def _test_step(self, data, target, dataset, index):
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        log_y_hat, _ = self.classification_module(data, index = index)
        
        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_current = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))

        return log_y_hat, neg_likelihood, mse_current


    def test(self,loader):
        self.classification_module.eval()

        dataset = loader.dataset
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_index, data in enumerate(loader):
                if len(data)==3 :
                    data, target, index = data
                else :
                    data, target = data
                    index = None

                log_y_hat, neg_likelihood, mse_current = self._test_step(data, target, dataset, index)
                
                test_loss += mse_current
                pred = log_y_hat.data.max(1, keepdim=True)[1]
                if self.use_cuda:
                    correct_current = pred.eq(target.cuda().data.view_as(pred)).sum()
                else :
                    correct_current = pred.eq(target.data.view_as(pred)).sum()
                correct += correct_current


        test_loss /= len(loader.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.test_loader.dataset),
        100. * correct / len(loader.test_loader.dataset)))
        total_dic = self._create_dic_test(correct/len(loader.test_loader.dataset), neg_likelihood, test_loss)
        return total_dic



class trainingWithSelection(ordinaryTraining):
    def __init__(self, classification_module,):
        super().__init__(classification_module, )

    def _train_step(self, data, target, dataset, index = None):
        self.zero_grad()
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)

        true_selection = dataset.get_true_selection(index = index, train_dataset = True)
        log_y_hat, _ = self.classification_module(data, sample_b = true_selection, index = index)

        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        loss = neg_likelihood
        dic = self._create_dic(loss, neg_likelihood, mse_loss)
        loss.backward()
        self.optim_classification.step()
        
        return dic

    def _test_step(self, data, target, dataset, index,):
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        true_selection = dataset.get_true_selection(index = index, train_dataset = False)
        log_y_hat, _ = self.classification_module(data, sample_b = true_selection, index = index)

        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_current = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        return log_y_hat, neg_likelihood, mse_current



class EVAL_X(ordinaryTraining):
    def __init__(self, classification_module, fixed_distribution, reshape_mask_function = None,):
        super().__init__(classification_module,)
        self.fixed_distribution = fixed_distribution
        self.reshape_mask_function = reshape_mask_function

    def train_epoch(self, epoch, loader, nb_sample_z = 10, save_dic=False, verbose=False,):
        self.train()
        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)

            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z = nb_sample_z)
            
            if batch_idx % 100 == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            print(f"Learning Rate classification : {self.scheduler_classification.get_last_lr()}")
            self.scheduler_classification.step()
        
        return total_dic

        

    def _train_step(self, data, target, dataset, index = None, nb_sample_z = 10):
        
        
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        batch_size = data.size(0)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = None)

        if index_expanded is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
        else :
            index_expanded_flatten = None

        z = self.fixed_distribution(nb_sample_z * batch_size)
        if self.reshape_mask_function is not None :
            z = self.reshape_mask_function(z)

        log_y_hat, _ = self.classification_module(data_expanded.flatten(0,1), z, index_expanded_flatten)
        log_y_hat = log_y_hat.reshape(nb_sample_z * batch_size, dataset.get_dim_output())

        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        loss = neg_likelihood
        dic = self._create_dic(loss, neg_likelihood, mse_loss)
        loss.backward()
        self.optim_classification.step()
        return dic
    

        
    def _test_step(self, data, target, dataset, index, nb_sample_z = 1):
        if self.use_cuda :
            data, _, index = on_cuda(data, target = None, index = index,)
        batch_size = data.size(0)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = None, index=index, one_hot_target = None, nb_sample_z = nb_sample_z, nb_imputation = None)

        if index_expanded is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
        else :
            index_expanded_flatten = None

        z = self.sampling_distribution(nb_sample_z * batch_size)


        log_y_hat, _ = self.classification_module(data_expanded.flatten(0,1), z, index_expanded_flatten)
        log_y_hat = log_y_hat.reshape(nb_sample_z*batch_size, -1)
        # log_y_hat_mean = torch.logsumexp(log_y_hat,0) - torch.log(torch.tensor(nb_sample_z))

        neg_likelihood = F.nll_loss(log_y_hat, target_expanded)
        mse_current = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target_expanded)**2,1))
        return log_y_hat, neg_likelihood, mse_current



class SELECTION_BASED_CLASSIFICATION(ordinaryTraining):
    """ Abstract class to help the classification of the different module """

    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False, post_hoc_guidance = None,
                argmax_post_hoc = False, show_variance_gradient = False, ):

        super().__init__(classification_module,)
        self.selection_module = selection_module
        self.distribution_module = distribution_module
        self.baseline = baseline


        self.reshape_mask_function = reshape_mask_function
        self.show_variance_gradient = show_variance_gradient
              
        self.fix_classifier_parameters = fix_classifier_parameters
        self.post_hoc_guidance = post_hoc_guidance
        self.argmax_post_hoc = argmax_post_hoc

        if self.fix_classifier_parameters :
            self.classification_module.fix_parameters()
            if self.need_feature :
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
 
    def reshape(self, z,):
        reshaped_z = self.reshape_mask_function(z)
        return reshaped_z

    def calculate_variance(self, loss):
        if self.show_variance_gradient :
            g = []
            for k in range(len(loss)):
                g2 = vectorize_gradient(torch.autograd.grad(loss[k], self.selection_module.destructor.parameters(), retain_graph = True, create_graph = False), 
                        self.selection_module.destructor.named_parameters(), set_none_to_zero=True)
                g.append((g2).unsqueeze(1))
            g_mean = torch.mean(torch.cat(g, axis=1),axis=1)
            g_square = torch.mean(torch.cat(g,axis=1)**2,axis=1)
            variance = torch.mean(g_square - g_mean**2)
        else :
            variance = None
        return variance

    def compile(self, optim_classification, optim_destruction, scheduler_classification = None, scheduler_destruction = None, optim_baseline = None, scheduler_baseline = None, optim_distribution_module = None, scheduler_distribution_module = None, **kwargs):
        self.optim_classification = optim_classification
        self.scheduler_classification = scheduler_classification
        self.optim_destruction = optim_destruction
        self.scheduler_destruction = scheduler_destruction
        self.compiled = True
        self.optim_distribution_module = optim_distribution_module
        self.scheduler_distribution_module = scheduler_distribution_module

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
            self.destruction_module.cuda()
            self.distribution_module.cuda()
            self.use_cuda = True

    def scheduler_step(self):
        assert(self.compiled)
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()
        if self.scheduler_destruction is not None :
            self.scheduler_destruction.step()
        if self.scheduler_distribution_module is not None :
            self.scheduler_distribution_module.step()
        self.distribution_module.update_distribution()

    def optim_step(self):
        assert(self.compiled)
        if not self.fix_classifier_parameters or self.optim_classification is None :
           self.optim_classification.step()
        self.optim_destruction.step()
        if self.optim_distribution_module is not None :
            self.optim_distribution_module.step()

    
    def train_epoch(self, epoch, loader, nb_sample_z = 10, save_dic=False, verbose=False):
        assert(self.compiled)
        self.train()
        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)

            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z = nb_sample_z)
            
            if batch_idx % 100 == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        self.scheduler_step()    
        return total_dic

    def _create_dic(self, loss_total, neg_likelihood, mse_loss, pi_list, loss_rec = None, loss_reg = None, loss_destruction = None, variance_gradient = None):
        dic = super()._create_dic(loss_total, neg_likelihood, mse_loss)

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
        dic["loss_destruction"] = get_item(loss_destruction)
        dic["variance_gradient"] = get_item(variance_gradient)
        return dic

    def _train_step(self, data, target, dataset, index = None, nb_sample_z = 10):
        raise NotImplementedError
    
    def _test_step(self, data, target, dataset, index, nb_sample_z = 1):
        raise NotImplementedError

    def test(self, loader, nb_sample_z = 10):
        test_loss_mse = 0
        test_loss_likelihood = 0
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
                data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = None)
                if index_expanded is not None :
                    index_expanded_flatten = index_expanded.flatten(0,1)
                else :
                    index_expanded_flatten = None



                log_pi_list, _ = self.selection_module(data)
                pi_list_total.append(torch.exp(log_pi_list).cpu().numpy())
                z = self.distribution_module(log_pi_list).sample((nb_sample_z,))
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

                ## Check the prediction with the destruction method
                log_y_hat_destructed, _ = self.classification_module(data_expanded.flatten(0,1), z, index = index_expanded_flatten)
                log_y_hat_destructed = log_y_hat_destructed.reshape(nb_sample_z, batch_size, loader.dataset.get_dim_output())
                log_y_hat_iwae = torch.logsumexp(log_y_hat_destructed,0)
                test_loss_likelihood += F.nll_loss(log_y_hat_iwae,target)
                test_loss_mse += torch.mean(torch.sum((torch.exp(log_y_hat_iwae)-one_hot_target)**2,1))

                pred_destructed = torch.argmax(log_y_hat_iwae, dim=1)
                correct_destructed += pred_destructed.eq(target).sum()



            test_loss_mse /= len(loader.test_loader.dataset) * batch_size
            print('\nTest set: MSE: {:.4f}, Likelihood {:.4f}, Accuracy No Destruction: {}/{} ({:.0f}%), Accuracy Destruction: {}/{} ({:.0f}%), Accuracy PostHoc: {}/{} ({:.0f}%),'.format(
                -test_loss_likelihood.item(), test_loss_mse.item(),
                 correct_classic.item(), len(loader.test_loader.dataset),100. * correct_classic.item() / len(loader.test_loader.dataset),
                 correct_destructed.item(), len(loader.test_loader.dataset), 100. * correct_destructed.item() / len(loader.test_loader.dataset),
                 correct_post_hoc, len(loader.test_loader.dataset), 100. * correct_post_hoc / len(loader.test_loader.dataset),
                ))
            print("\n")
            total_dic = self._create_dic_test(correct_destructed/len(loader.test_loader.dataset),
                correct_classic/len(loader.test_loader.dataset),
                test_loss_likelihood,
                test_loss_mse,
                pi_list_total,
                correct_post_hoc=correct_post_hoc)

        return total_dic

    def _create_dic_test(self, correct, correct_no_destruction, neg_likelihood, test_loss, pi_list_total, correct_post_hoc = None):
        total_dic = super()._create_dic_test(correct_no_destruction, neg_likelihood, test_loss)
        treated_pi_list_total = np.concatenate(pi_list_total)
        total_dic["mean_pi_list"] = np.mean(treated_pi_list_total).item()
        total_dic["accuracy_prediction_destruction"] = correct.item()
        q = np.quantile(treated_pi_list_total, [0.25,0.5,0.75])
        total_dic["pi_list_q1"] = q[0].item()
        total_dic["pi_list_median"] = q[1].item()
        total_dic["pi_list_q2"] = q[2].item()
        total_dic["correct_post_hoc"] = correct_post_hoc

        return total_dic


    def MCMC(self, dataset, data, target, sampling_distribution, Niter, index = None, nb_sample_z = 1,  eps = 1e-6, burn = 1000, jump = 50, return_pred = False):
        self.eval()
        with torch.no_grad():
            sample_list = []
            sample_list_readable = []
            y_hat_list = []
            data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten, index_expanded = prepare_data_augmented(data, target, index = index, num_classes=dataset.get_dim_output(), nb_sample_z=nb_sample_z, use_cuda=self.use_cuda)


            pi_list, log_pi_list,  _, previous_z, pz = self._destructive_test(data, sampling_distribution, nb_sample_z)
            log_y_hat, _ = self.classification_module(data_expanded_flatten, previous_z, index_expanded)
            log_y_hat_iwae = torch.logsumexp(log_y_hat, 0)
            previous_log_py = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)

            for k in range(Niter):
                z, p_z = self._sample_z_test(pi_list, sampling_distribution, nb_sample_z)

                log_y_hat, _  = self.classification_module(data_expanded_flatten, z, index_expanded)
                log_y_hat = log_y_hat.reshape(nb_sample_z, -1, dataset.get_dim_output())
                log_y_hat_iwae = torch.logsumexp(log_y_hat, 0)

                log_py = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)


                u = torch.rand((log_py.shape)) 
                if self.use_cuda:
                    u = u.cuda()

                proba_acceptance = torch.exp(log_py-previous_log_py)
                mask_acceptance = u<proba_acceptance
                mask_acceptance = mask_acceptance.unsqueeze(1).expand((-1,z.shape[-1]))
                previous_z = torch.where(mask_acceptance, z, previous_z)
                log_y_hat, _  = self.classification_module(data_expanded_flatten, previous_z, index_expanded)
                log_y_hat = log_y_hat.reshape(nb_sample_z, -1, dataset.get_dim_output())
                log_y_hat_iwae = torch.logsumexp(log_y_hat, 0)
                previous_log_py = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)

                if k > burn and k%jump == 0 :
                    sample_list.append(previous_z.cpu()[None,:,:])
                    sample_list_readable.append(self.classification_module.readable_sample(previous_z).cpu()[None, :, :])

            sample_list_readable = torch.mean(torch.cat(sample_list_readable),0)
            sample_list = torch.mean(torch.cat(sample_list),0)

            if return_pred :
                return sample_list_readable, log_y_hat_iwae

            return sample_list_readable




class REALX(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, distribution_module,
                classification_distribution, baseline = None, reshape_mask_function = None, fix_classifier_parameters = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,):
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters,
                        post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc,
                        show_variance_gradient = show_variance_gradient)

        self.classification_distribution = classification_distribution

    def _train_step(self, data, target, dataset, index = None, nb_sample_z = 10):

        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        batch_size = data.size(0)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = None)

        if index_expanded is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
        else :
            index_expanded_flatten = None

        z = self.fixed_distribution(nb_sample_z * batch_size)
        z = self.reshape(z)

        log_y_hat, _ = self.classification_module(data_expanded.flatten(0,1), z, index_expanded_flatten)
        log_y_hat = log_y_hat.reshape(nb_sample_z * batch_size, dataset.get_dim_output())

        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        loss = neg_likelihood
        dic = self._create_dic(loss, neg_likelihood, mse_loss,)
        loss.backward()
        self.optim_classification.step()
        return dic

class ReparametrizedTraining(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, distribution_module, baseline = None, reshape_mask_function = None, fix_classifier_parameters = False, post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,):
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline, reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters, post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc, show_variance_gradient = show_variance_gradient)


    def _train_step(self, data, target, dataset, index = None,  nb_sample_z = 10,):
        self.zero_grad()
        
        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,1)
        else :
            index_expanded_flatten = None
        
        # Destructive module :
        log_pi_list, loss_reg = self.selection_module(data)

        # Distribution :
        p_z = self.distribution_module(log_pi_list)
        z = self.distribution_module.sample((nb_sample_z,))
        z = self.reshape(z)

        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,1), z, index_expanded_flatten)


        
        # Loss for classification
        if self.post_hoc_guidance is None :
            neg_likelihood = F.nll_loss(log_y_hat, target_expanded_multiple_imputation.flatten(0,2), reduce = False).reshape(nb_imputation, nb_sample_z, batch_size)
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target_expanded_multiple_imputation.flatten(0,2))**2,1)) 
        else :
            out_y, _ = self.post_hoc_guidance(data_expanded_multiple_imputation.flatten(0,2), index = index_expanded_multiple_imputation_flatten)
            out_y = out_y.detach().detach()
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-torch.exp(out_y).float())**2,1)) 
            if self.argmax_post_hoc :
                out_y = torch.argmax(out_y, -1)
                neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce = False)
            else :
                neg_likelihood = - torch.sum(torch.exp(out_y) * log_y_hat, -1).reshape(nb_imputation, nb_sample_z, batch_size)


        # Loss for selection
        neg_likelihood = torch.logsumexp(neg_likelihood,0) - torch.log(torch.tensor(nb_imputation, dtype=torch.float32))
        neg_likelihood = torch.logsumexp(neg_likelihood,0) - torch.log(torch.tensor(nb_sample_z, dtype=torch.float32))
        neg_likelihood = torch.sum(neg_likelihood)

        # Updates 
        loss_rec = neg_likelihood
        loss_total = loss_rec + loss_reg + loss_reconstruction
        loss_total.backward()
        self.optim_step()

        # Measures :
        dic = self._create_dic(loss_total, neg_likelihood, mse_loss,  loss_rec = loss_rec, loss_reg = loss_reg, pi_list = torch.exp(log_pi_list))

        return dic


        

class AllZTraining(SELECTION_BASED_CLASSIFICATION):
    """ Difference betzeen the previous is the way we calculate the multiplication with the loss"""
    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,):
        
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters,
                        post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc, show_variance_gradient = show_variance_gradient)
        self.z = None
        self.computed_combination = False


    def _train_step(self, data, target, dataset, index = None,  nb_sample_z = None,):
        self.zero_grad()
        dim_total = np.prod(data.shape[1:])
        batch_size = data.shape[0]
        nb_sample_z = 2**dim_total
        nb_imputation = self.classification_module.imputation.nb_imputation

        # Create all z combinations :
        if not self.computed_combination :
            self.z = get_all_z(dim_total)
            self.computed_combination = True
        z = self.z.unsqueeze(1).expand(nb_sample_z, batch_size, dim_total).detach()
        if self.use_cuda:
            z = z.cuda()
        
        # Prepare data :
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,1)
        
        
        # Destructive module :
        log_pi_list, loss_reg = self.selection_module(data)
        # Distribution :
        p_z = self.distribution_module(log_pi_list)


        # z = p_z.sample((nb_sample_z,))
        log_prob_pz = torch.sum(p_z.log_prob(z.flatten(2)), axis = -1) # TODO : torch.logsumexp ou torch.Sum ? It's supposed to be sum as we want the product of the bernoulli hence, the sum of the log bernoulli
        z = self.reshape(z)
        
        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,1), z, index = index_expanded_flatten)
    

        # Loss for classification:
        if self.post_hoc_guidance is None :
            neg_likelihood = F.nll_loss(log_y_hat, target_expanded_multiple_imputation.flatten(0,2), reduce=False).reshape(nb_imputation, nb_sample_z, batch_size)
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target_expanded_multiple_imputation.flatten(0,2))**2,1))
        else :
            out_y, _ = self.post_hoc_guidance(data_expanded_multiple_imputation.flatten(0,2), index = index_expanded_multiple_imputation_flatten)
            out_y = out_y.detach()
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-torch.exp(out_y).float())**2,1)) 
            if self.argmax_post_hoc :
                out_y = torch.argmax(out_y, -1)
                neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce=False).reshape(nb_imputation, nb_sample_z, batch_size)
            else :
                neg_likelihood = -torch.sum(torch.exp(out_y) * log_y_hat, axis=-1).reshape(nb_imputation, nb_sample_z, batch_size)

        # Loss for selection :
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32))
        neg_likelihood *= torch.exp(log_prob_pz) # I used to do + but not correct ?
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_sample_z).type(torch.float32))
        neg_likelihood = torch.sum(neg_likelihood)


        # Update :
        loss_total = neg_likelihood + loss_reg
        loss_total.backward()
        self.optim_step()
        

        # Measures :
        dic = self._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec = loss_reconstruction, loss_reg = loss_reg, pi_list = torch.exp(log_pi_list))

        return dic



class REINFORCE(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,  ):
        
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters,
                        post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc, show_variance_gradient=show_variance_gradient,)



    def _train_step(self, data, target, dataset, index = None,  nb_sample_z = 10,):

        self.zero_grad()
        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]

        # Get data :
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z = nb_sample_z, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,1)
        else :
            index_expanded_flatten = None
            index_expanded_multiple_imputation_flatten = None
        # Prepare baseline :
        if self.baseline is not None:
            log_y_hat_baseline = self.baseline(data)
            log_y_hat_baseline_masked = torch.masked_select(log_y_hat_baseline, one_hot_target>0.5)
            loss_baseline = F.nll_loss(log_y_hat_baseline, target, reduce = False) # Batch_size, 1

        # Selection Module :
        log_pi_list, loss_reg = self.selection_module(data)


        # Distribution :

        p_z = self.distribution_module(log_pi_list)
        z = p_z.sample((nb_sample_z,))
        log_prob_pz = torch.sum(p_z.log_prob(z).flatten(2), axis = -1) # TODO : torch.logsumexp ou torch.Sum ? It's supposed to be sum as we want the product of the bernoulli hence, the sum of the log bernoulli
        z = self.reshape(z)


        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,1), z.detach(), index = index_expanded_flatten)
        
        # Choice for target :
        if self.post_hoc_guidance is None :
            neg_likelihood = F.nll_loss(log_y_hat, target_expanded_multiple_imputation.flatten(), reduce=False).reshape(nb_imputation, nb_sample_z, batch_size)
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target_expanded_multiple_imputation.flatten(0,2))**2,1))
        else :
            out_y, _ = self.post_hoc_guidance(data_expanded_multiple_imputation.flatten(0,2), index = index_expanded_multiple_imputation_flatten)
            out_y = out_y.detach().detach()
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-torch.exp(out_y).float())**2,1)) 
            if self.argmax_post_hoc :
                out_y = torch.argmax(out_y, -1)
                neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce=False).reshape(nb_imputation, nb_sample_z, batch_size)
            else :
                neg_likelihood = -torch.sum(torch.exp(out_y) * log_y_hat, axis=-1).reshape(nb_imputation, nb_sample_z, batch_size)
        neg_likelihood_original = neg_likelihood.clone().detach()

        # Loss for classification
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32))
        # neg_likelihood *= torch.exp(log_prob_pz).detach()
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_sample_z).type(torch.float32))
        neg_likelihood = torch.sum(neg_likelihood)
        # print(neg_likelihood)
        loss_classification_module = neg_likelihood

        # Loss for selection :
        neg_reward = neg_likelihood_original
        if self.baseline is not None :
            neg_reward = neg_reward - log_y_hat_baseline_masked.unsqueeze(0).unsqueeze(1).expand(nb_imputation, nb_sample_z, batch_size).detach()

        neg_reward = torch.logsumexp(neg_reward, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32))
        loss_hard =  log_prob_pz*neg_reward
        loss_destruction = torch.logsumexp(loss_hard, axis=0) - torch.log(torch.tensor(nb_sample_z).type(torch.float32))
        loss_destruction = torch.sum(loss_destruction) 


        # Update :
        if self.baseline is not None :
            loss_total = loss_destruction + loss_classification_module + loss_reg + loss_reconstruction + loss_baseline
        else :
            loss_total = loss_destruction + loss_classification_module + loss_reg + loss_reconstruction
        
        variance_gradient = self.calculate_variance(loss_hard,)
        loss_total.backward()
        self.optim_step()

        dic = self._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec = loss_reconstruction, loss_reg = loss_reg, pi_list = torch.exp(log_pi_list), loss_destruction = loss_destruction, variance_gradient = variance_gradient)

        return dic


## FOR THE MOMENT, NO LOOK AT THAT :


class REBAR(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, baseline = None, feature_extractor = None, kernel_patch = (1,1), stride_patch = (1,1), fix_classifier_parameters = False, post_hoc_guidance = None, argmax_post_hoc_classification = False, feature_extractor_training = False, use_cuda = True, update_temperature = False, pytorch_relax = False):
        super().__init__(classification_module, selection_module,baseline=baseline, feature_extractor= feature_extractor, kernel_patch = kernel_patch, stride_patch = stride_patch, use_cuda=use_cuda, fix_classifier_parameters=fix_classifier_parameters, post_hoc_guidance=post_hoc_guidance, argmax_post_hoc_classification=argmax_post_hoc_classification)
        self.compteur = 0
        self.update_temperature = update_temperature
        self.pytorch_relax = pytorch_relax
        self.eta_classif = {}
        self.eta_destruction = {}
        # self._initialize_eta(self.eta_classif, self.classification_module.classifier.named_parameters())
        self._initialize_eta(self.eta_destruction, self.selection_module.destructor.named_parameters())
        self.temperature_lambda = torch.tensor(0., requires_grad = self.update_temperature, device='cuda:0')
        self.create_optimizer()

    def zero_grad(self):
        super().zero_grad()
        for name in self.eta_destruction.keys():
            if self.eta_destruction[name].grad is not None :
                self.eta_destruction[name].grad.zero_()
        if self.temperature_lambda.grad is not None :
            self.temperature_lambda.grad.zero_()
        # for name in self.eta_classif.keys():
            # if self.eta_classif[name].grad is not None :
                # self.eta_classif[name].grad.zero_()

    def _destructive_train(self, data, sampling_distribution, nb_sample_z):
        pi_list, loss_reg = self._get_pi(data)
        if (pi_list<0).any() or torch.isnan(pi_list).any() or torch.isinf(pi_list).any():
            print(pi_list)
        assert((pi_list>=0).any())
        assert((pi_list<=1).any())
        z, p_z = self._sample_z_test(pi_list, sampling_distribution, nb_sample_z)
        return pi_list, loss_reg, z, p_z


    def create_optimizer(self, lr = 1e-4):
        list = []
        for name in self.eta_destruction:
            list.append(self.eta_destruction[name])
        if self.update_temperature :
            list.append(self.temperature_lambda)
        self.optimizer_eta = torch.optim.SGD(list, lr=lr)


    def _multiply_by_eta_per_layer(self, parameters, eta):
        for (name, p) in parameters:
            if p.grad is None :
                continue
            else :
                p.grad = p.grad * eta[name]

    def _multiply_by_eta_per_layer_gradient(self, gradients, eta, named_parameters):
        g_dic = {}
        for g,(name, _) in zip(gradients,named_parameters):
            if g is None :
                g_dic[name] = None
            else :
                g_dic[name] = g * eta[name]
        return g_dic


    def _initialize_eta(self, eta, parameters):
        for (name, _) in parameters:
            if name in eta.keys() :
                print("Doubling parameters here")
            if self.use_cuda :
                eta[name] = torch.rand(1, requires_grad=True, device="cuda:0") # TODO
                # eta[name] = torch.ones(1, requires_grad=True, device='cuda:0')
            else :
                eta[name] = torch.rand(1, requires_grad=True, device="cpu") # TODO


    def _create_dic(self, loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list, loss_destruction = None, variance = None):
        dic = super()._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list, loss_destruction)
        if variance is not None :
            dic["variance_grad"] = variance.detach().cpu().item()
        else :
            dic["variance_grad"] = variance
        if self.update_temperature :
            dic["temperature"] = torch.exp(self.temperature_lambda).detach().cpu().item()

        eta_mean = 0
        for k, key in enumerate(self.eta_destruction.keys()):
            if k == 0 :
                eta_mean = self.eta_destruction[key].cpu().detach().item()
            eta_mean = (eta_mean * k + self.eta_destruction[key].cpu().detach().item())/ (k+1)
        dic["eta_mean"] = eta_mean

        return dic

    def set_variance_grad(self, network):
        grad = torch.nn.utils.parameters_to_vector(-p.grad for p in network.parameters()).flatten()
        variance = torch.mean(grad**2)
        return variance



    def _train_step(self, data, target, dataset,  optim_classifier, optim_destruction, sampling_distribution, index = None, optim_baseline = None, optim_feature_extractor =None, lambda_reg = 0.0, nb_sample_z = 10, lambda_reconstruction = 0.0):
        
        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten, index_expanded = prepare_data_augmented(data, target, index = index, num_classes=dataset.get_dim_output(), nb_sample_z=nb_sample_z, use_cuda=self.use_cuda)

        self.zero_grad()
        # Selection module :
        pi_list, loss_reg, z_real_sampled, p_z = self._destructive_train(data, sampling_distribution, nb_sample_z)
        loss_reg = lambda_reg * loss_reg
        
        pi_list_extended = torch.cat([pi_list for k in range(nb_sample_z)], axis=0)
        # u = torch.FloatTensor(1., batch_size * nb_imputation,  requires_grad=False).uniform_(0, 1) + 1e-9
        if self.use_cuda :
            u = (torch.rand((z_real_sampled.shape), requires_grad = False).flatten(0,1) + 1e-9).clamp(0,1).cuda() # TODO
            v = (torch.rand((z_real_sampled.shape), requires_grad = False).flatten(0,1) + 1e-9).clamp(0,1).cuda()
        else :
            u = (torch.rand((z_real_sampled.shape), requires_grad = False).flatten(0,1) + 1e-9).clamp(0,1)
            v = (torch.rand((z_real_sampled.shape), requires_grad = False).flatten(0,1) + 1e-9).clamp(0,1)



        b = reparam_pz(u, pi_list_extended)
        z = Heaviside(b)
        tilde_b = reparam_pz_b(v, z, pi_list_extended)


        soft_concrete_rebar_z = sigma_lambda(b, torch.exp(self.temperature_lambda))  
        if not self.pytorch_relax :
            soft_concrete_rebar_tilde_z = sigma_lambda(tilde_b, torch.exp(self.temperature_lambda))
        else :
            distrib = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.exp(self.temperature_lambda), probs = pi_list_extended)
            soft_concrete_rebar_tilde_z = distrib.rsample()


        # Classification module hard :
        log_y_hat, _ = self.classification_module(data_expanded_flatten, z.detach(), index_expanded)
        log_y_hat = log_y_hat.reshape(nb_sample_z, nb_imputation, batch_size, dataset.get_dim_output())
        log_y_hat_iwae = torch.logsumexp(log_y_hat,0) + torch.log(torch.tensor(1./nb_sample_z))
        log_y_hat_iwae = torch.mean(log_y_hat_iwae,0)
        # log_y_hat_iwae = torch.logsumexp(torch.logsumexp(log_y_hat,1),0) + torch.log(torch.tensor(1./nb_imputation))+ torch.log(torch.tensor(1./nb_sample_z)) # Need verification of this with the masked version
        log_y_hat_iwae_masked_hard_z = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)

        # Classification module relaxed:
        log_y_hat, _ = self.classification_module(data_expanded_flatten,  soft_concrete_rebar_tilde_z, index_expanded)
        log_y_hat = log_y_hat.reshape(nb_sample_z, nb_imputation, batch_size, dataset.get_dim_output())
        log_y_hat_iwae = torch.logsumexp(log_y_hat,0) + torch.log(torch.tensor(1./nb_sample_z))
        log_y_hat_iwae = torch.mean(log_y_hat_iwae,0)
        # log_y_hat_iwae = torch.logsumexp(torch.logsumexp(log_y_hat,1),0) + torch.log(torch.tensor(1./nb_imputation))+ torch.log(torch.tensor(1./nb_sample_z)) # Need verification of this with the masked version
        log_y_hat_iwae_masked_soft_tilde_z = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)


        # Classification module relaxed_2 :
        log_y_hat, _ = self.classification_module(data_expanded_flatten, soft_concrete_rebar_z, index_expanded)
        log_y_hat = log_y_hat.reshape(nb_sample_z, nb_imputation, batch_size, dataset.get_dim_output())
        log_y_hat_iwae = torch.logsumexp(log_y_hat,0) + torch.log(torch.tensor(1./nb_sample_z))
        log_y_hat_iwae = torch.mean(log_y_hat_iwae,0)
        # log_y_hat_iwae = torch.logsumexp(torch.logsumexp(log_y_hat,1),0) + torch.log(torch.tensor(1./nb_imputation))+ torch.log(torch.tensor(1./nb_sample_z)) # Need verification of this with the masked version
        log_y_hat_iwae_masked_soft_z = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)

        # Loss calculation
        z = z.reshape(z_real_sampled.shape)
        log_prob_z_sampled = p_z.log_prob(z.detach()).reshape(nb_sample_z, batch_size, -1)
        log_prob_z = torch.sum(torch.sum(log_prob_z_sampled,axis=0), axis=-1) 

        likelihood_hard = log_y_hat_iwae_masked_hard_z.detach() * log_prob_z + log_y_hat_iwae_masked_hard_z
        loss_hard = - likelihood_hard
        likelihood_control_variate = - log_y_hat_iwae_masked_soft_tilde_z.detach() * log_prob_z + log_y_hat_iwae_masked_soft_z - log_y_hat_iwae_masked_soft_tilde_z
        loss_control_variate = - likelihood_control_variate


        g = []
        for k in range(len(loss_hard)):
            g1 = torch.autograd.grad(loss_control_variate[k], self.selection_module.destructor.parameters(), retain_graph = True, create_graph = False)
            g2 = vectorize_gradient(torch.autograd.grad(loss_hard[k], self.selection_module.destructor.parameters(), retain_graph = True, create_graph = False), 
                    self.selection_module.destructor.named_parameters(), set_none_to_zero=True)
            g1 = self._multiply_by_eta_per_layer_gradient(g1, self.eta_destruction, self.selection_module.destructor.named_parameters())
            g1= vectorize_dic(g1, self.selection_module.destructor.named_parameters(), set_none_to_zero=True)
            g.append((g2+g1).unsqueeze(1))
        g_mean = torch.mean(torch.cat(g, axis=1),axis=1)
        g_square = torch.mean(torch.cat(g,axis=1)**2,axis=1)
        variance = torch.mean(g_square - g_mean**2)
        
        torch.autograd.backward(variance, inputs = [self.eta_destruction[name] for name in self.eta_destruction.keys()])
        if self.update_temperature :
            torch.autograd.backward(variance, inputs = [self.temperature_lambda], retain_graph = True)


        self.classification_module.classifier.zero_grad()
        self.selection_module.destructor.zero_grad()

        torch.mean(loss_control_variate).backward(retain_graph = True, create_graph = False)
        self._multiply_by_eta_per_layer(self.selection_module.destructor.named_parameters(), self.eta_destruction)
        self.classification_module.classifier.zero_grad()
        torch.mean(loss_hard).backward(retain_graph = True, create_graph = False)

        optim_destruction.step()
        optim_classifier.step()
        self.optimizer_eta.step()

        
        # # Temperature T optimisation :
        # gumbel_learning_signal = - log_y_hat_iwae_masked_soft_tilde_z
        # df_dt = []
        # for k in range(len(gumbel_learning_signal)):
        #     df_dt.append(torch.nn.utils.parameters_to_vector(torch.autograd.grad(gumbel_learning_signal[k], self.temperature_lambda, retain_graph=True)))
        # df_dt = torch.cat(df_dt)
        # backward_aux = torch.autograd.grad(torch.mean(df_dt.detach() * log_prob_z),self.selection_module.destructor.parameters(), retain_graph=True)
        # reinf_g_t = self._multiply_by_eta_per_layer_gradient(
        #     backward_aux,
        #     self.eta_destruction,
        #     self.selection_module.destructor.named_parameters() 
        # )
        # reinf_g_t = vectorize_dic(reinf_g_t, self.selection_module.destructor.named_parameters(), set_none_to_zero=True)

        # reparam = log_y_hat_iwae_masked_soft_z - log_y_hat_iwae_masked_soft_tilde_z
        # reparam_g = torch.autograd.grad(torch.mean(reparam), self.selection_module.destructor.parameters(), retain_graph = True, create_graph = True)
        # reparam_g = self._multiply_by_eta_per_layer_gradient(reparam_g, self.eta_destruction, self.selection_module.destructor.named_parameters())
        # reparam_g = vectorize_dic(reparam_g, self.selection_module.destructor.named_parameters(), set_none_to_zero=True)
        # g_aux = torch.nn.utils.parameters_to_vector(-p.grad for p in self.selection_module.destructor.parameters())
        # reparam_g_t = torch.autograd.grad(torch.mean(2*g_aux*reparam_g), self.temperature_lambda, retain_graph = True)[0]


        # grad_t = torch.mean(2*g_aux*reinf_g_t) + reparam_g_t
        # self.temperature_lambda = self.temperature_lambda - 1e-4 * grad_t 
        # self.temperature_lambda.backward()
        # self.temperature_lambda.grad.zero_()
        
        loss_total = torch.mean(loss_hard + loss_control_variate)

        dic = self._create_dic(loss_total, -log_y_hat_iwae_masked_hard_z.mean(), torch.tensor(0.), torch.tensor(0.), loss_reg, pi_list, torch.mean(loss_hard), variance)

        return dic


class variationalTraining(ReparametrizedTraining):
    def __init__(self, classification_module, selection_module, selection_module_var, baseline = None, feature_extractor = None, kernel_patch = (1,1), stride_patch = (1,1)):
        super().__init__(classification_module, selection_module,baseline = baseline, feature_extractor = feature_extractor, kernel_patch = kernel_patch, stride_patch = stride_patch)
        self.selection_module_var = selection_module_var
        if self.use_cuda :
            self.selection_module_var.cuda()

    
    def zero_grad(self):
        super().zero_grad()
        self.selection_module.zero_grad()

    def eval(self):
        super().eval()
        self.selection_module.eval()

    def train(self):
        super().train()
        self.selection_module.train()

    
    def _prob_calc(self, y_hat, one_hot_target_expanded, z , pz, qz):
        nb_sample_z = one_hot_target_expanded.shape[0]
        log_prob_y = torch.masked_select(y_hat,one_hot_target_expanded>0.5).reshape(nb_sample_z,-1)
        log_prob_pz = torch.sum(pz.log_prob(z),-1)
        log_prob_qz = torch.sum(qz.log_prob(z),-1)
        
        return log_prob_y,log_prob_pz,log_prob_qz
    
    def _likelihood_var(self, y_hat, one_hot_target_expanded, z, pz, qz):
        log_prob_y, log_prob_pz, log_prob_qz = self._prob_calc(y_hat, one_hot_target_expanded, z, pz, qz)
        return torch.mean(torch.logsumexp(log_prob_y+log_prob_pz-log_prob_qz,0))


    def _train_step(self, data, target, dataset, sampling_distribution, sampling_distribution_var, optim_classifier, optim_destruction, optim_destruction_var, index = None, optim_feature_extractor = None, optim_baseline = None, lambda_reg = 0.0, lambda_reg_var= 0.0, nb_sample_z = 10, lambda_reconstruction = 0.0):
        self.zero_grad()
        batch_size = data.shape[0]
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten, index_expanded = prepare_data_augmented(data, target, index = index, num_classes=dataset.get_dim_output(), nb_sample_z=nb_sample_z, use_cuda=self.use_cuda)

        pi_list, log_pi_list, loss_reg = self.selection_module(data)
        pi_list_var, log_pi_list_var, loss_reg_var = self.selection_module_var(data, one_hot_target = one_hot_target)
        loss_reg = lambda_reg * loss_reg
        loss_reg_var = lambda_reg_var * loss_reg_var

        pz = sampling_distribution(probs = pi_list)
        qz = sampling_distribution_var(pi_list_var)
        
        
        z = qz.rsample((nb_sample_z,))
        
        y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z, index = None,)
        y_hat = y_hat.reshape((nb_sample_z,batch_size, dataset.get_dim_output()))
        loss_reconstruction = lambda_reconstruction * loss_reconstruction
        y_hat_mean = torch.mean(y_hat, 0) # TODO CHANGE THIS


        neg_likelihood = - self._likelihood_var(y_hat,one_hot_target_expanded, z, pz, qz)
        mse_loss = torch.mean(torch.sum((torch.exp(y_hat_mean)-one_hot_target)**2,1))
        loss_rec = neg_likelihood
        loss_total = loss_rec + loss_reg + loss_reg_var + loss_reconstruction
        loss_total.backward()

        dic = self._create_dic(
            loss_total,
            neg_likelihood, mse_loss, loss_rec,
            loss_reg, pi_list,
            loss_reg_var, pi_list_var
            )

        optim_classifier.step()
        optim_destruction.step()
        optim_destruction_var.step()


        return dic
        

    def _create_dic(self,loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list, loss_reg_var, pi_list_var):
        dic = super()._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list)
        dic["loss_reg_var"] = loss_reg_var.item()
        dic["mean_pi_var"] = torch.mean((torch.mean(pi_list_var.squeeze(),1))).item()
        quantiles = torch.tensor([0.25,0.5,0.75])
        if self.use_cuda: 
            quantiles = quantiles.cuda()
        q = torch.quantile(pi_list_var.flatten(1),quantiles,dim=1,keepdim = True)
        dic["median_pi_var"] = torch.mean(q[1]).item()
        dic["q1_pi_var"] = torch.mean(q[0]).item()
        dic["q2_pi_var"] = torch.mean(q[2]).item()
        dic["mean pi diff"] = torch.mean(
            torch.mean(
                (pi_list.flatten(1)-pi_list_var.flatten(1))**2,
                1
                )
            ).item()
        return dic

    def _create_dic_test(self, correct, neg_likelihood, test_loss, pi_list_total, pi_list_var_total=None):
        total_dic = super()._create_dic_test(correct, neg_likelihood, test_loss, pi_list_total)
        if pi_list_var_total is None :
            return total_dic
        else :
            treated_pi_list_var_total = np.concatenate(pi_list_var_total)
            total_dic["mean_pi_list_var"] = np.mean(treated_pi_list_var_total)
            q = np.quantile(treated_pi_list_var_total, [0.25,0.5,0.75])
            total_dic["pi_list_var_q1"] = q[0]
            total_dic["pi_list_var_median"] = q[1]
            total_dic["pi_list_var_q2"] = q[2]
            
            return total_dic

    def train_epoch(self, epoch, dataset, optim_classifier, optim_destruction,
                optim_destruction_var, sampling_distribution, sampling_distribution_var,
                optim_baseline = None, optim_feature_extractor = None,
                lambda_reg = 0.0, lambda_reg_var= 0.0, nb_sample_z = 10,
                lambda_reconstruction = 0.0, save_dic = False, print_dic_bool = False,
                scheduler_classification = None, scheduler_destruction = None, scheduler_destruction_var = None,
                scheduler_baseline = None, scheduler_feature_extractor = None, ):
        self.train()
        total_dic = {}
        for batch_idx, data_aux in enumerate(dataset.train_loader):
            if self.give_index:
                data, target, index = data_aux
            else :
                data, target = data_aux
                index = None

            dic = self._train_step(
                data, target, dataset,
                sampling_distribution, sampling_distribution_var,
                optim_classifier,
                optim_destruction,
                optim_destruction_var,
                index = index,
                optim_baseline = optim_baseline,
                optim_feature_extractor = optim_feature_extractor,
                lambda_reg = lambda_reg, lambda_reg_var= lambda_reg_var,
                nb_sample_z = nb_sample_z, lambda_reconstruction=lambda_reconstruction
                )
                    
            if batch_idx % 100 == 0:
                if print_dic_bool:
                    print_dic(epoch, batch_idx, dic, dataset)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

        if scheduler_classification is not None :
            print(f"Learning Rate classification : {scheduler_classification.get_last_lr()}")
            scheduler_classification.step()
        if scheduler_baseline is not None :
            scheduler_baseline.step()
        if scheduler_destruction is not None :
            print(f"Learning Rate destruction : {scheduler_destruction.get_last_lr()}")
            scheduler_destruction.step()
        if scheduler_feature_extractor is not None :
            scheduler_feature_extractor.step()
        if scheduler_destruction_var is not None :
            scheduler_destruction_var.step()

        return total_dic


 

            
    def test_var(self, dataset, sampling_distribution, sampling_distribution_var, nb_sample_z = 10):
        self.eval()


        test_loss_mse = 0
        test_loss_likelihood = 0
        correct = 0
        
        
        with torch.no_grad():
            pi_list_total = []
            pi_list_var_total = []
            for aux in loader.test_loader:
                if self.give_index :
                    data, target, index = aux
                else :
                    data, target = aux
                    index = None
                    
                    
                batch_size = data.shape[0]
                data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten, index_expanded = prepare_data_augmented(data, target, index = index, num_classes=dataset.get_dim_output(), nb_sample_z=nb_sample_z, use_cuda=self.use_cuda)

                pi_list, _= self.selection_module(data, test=True)
                pi_list_var, _ = self.selection_module_var(data, one_hot_target=one_hot_target, test=True)
                pi_list_total.append(pi_list.cpu().numpy())
                pi_list_var_total.append(pi_list_var.cpu().numpy())

                pz = sampling_distribution(probs = pi_list)
                qz = sampling_distribution_var(pi_list_var)

                z = qz.sample()

                y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z, index_expanded)
                y_hat_squeeze = y_hat.squeeze()

                test_loss_likelihood = self._likelihood_var(y_hat,one_hot_target_expanded,z,pz,qz)
                test_loss_mse += torch.sum(torch.sum((torch.exp(y_hat_squeeze)-one_hot_target)**2,1))
                pred = torch.argmax(y_hat,dim = 1)
                correct += pred.eq(target).sum()

            test_loss_mse /= len(loader.test_loader.dataset) * dataset.batch_size_test
            print('\nTest set: AMSE: {:.4f}, Likelihood {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss_likelihood, test_loss_mse, correct, len(loader.test_loader.dataset),
                100. * correct / len(loader.test_loader.dataset)))
            print("\n")
            total_dic = self._create_dic_test(correct/len(loader.test_loader.dataset),
                test_loss_likelihood,
                test_loss_mse,
                pi_list_total,
                pi_list_var_total)

            return total_dic




    def MCMC_var(self, dataset, data, target, sampling_distribution, sampling_distribution_var, Niter, eps = 1e-6, burn = 1000, jump = 50, return_pred = False):
        self.classification_module.eval()
        self.selection_module.eval()
        with torch.no_grad():
            sample_list = []
            sample_list_readable = []

            data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten, index_expanded = prepare_data_augmented(data, target, index = index, num_classes=dataset.get_dim_output(), nb_sample_z=nb_sample_z, use_cuda=self.use_cuda)
            batch_size = data.shape[0]
            input_size = data.shape[1] * data.shape[2] * data.shape[3]

            pi_list, _, _, _ = self.selection_module(data, test=True)
            pi_list_var, _, _, _ = self.selection_module_var(data, one_hot_target=one_hot_target, test=True)
            pz = sampling_distribution(probs = pi_list)
            qz = sampling_distribution_var(pi_list_var)
            previous_z = qz.sample()
     

            y_hat, _  = self.classification_module(data_expanded_flatten, previous_z)
            previous_log_py, previous_log_pz, previous_log_qz = self._prob_calc(y_hat, one_hot_target_expanded, previous_z, pz, qz)

            for k in range(Niter):
                z = qz.sample()

                y_hat, _ = self.classification_module(data_expanded_flatten, z)
                log_py, log_pz, log_qz = self._prob_calc(y_hat, one_hot_target_expanded, z, pz, qz)

                u = torch.rand((batch_size)) 
                if self.use_cuda:
                    u = u.cuda()
                

                proba_acceptance = torch.exp((log_py + log_pz - log_qz) - (previous_log_py + previous_log_pz - previous_log_qz)).squeeze()
                mask_acceptance = u<proba_acceptance
                mask_acceptance = mask_acceptance.unsqueeze(1).expand((batch_size,z.shape[1]))
                previous_z = torch.where(mask_acceptance, z, previous_z)
                y_hat, _  = self.classification_module(data_expanded_flatten, previous_z)
                y_hat_squeeze = y_hat.squeeze()
                previous_log_py, previous_log_pz, previous_log_qz = self._prob_calc(y_hat, one_hot_target_expanded, previous_z, pz, qz)

                if k > burn and k%jump == 0 :
                    sample_list.append(previous_z.cpu()[None,:,:])
                    sample_list_readable.append(self.classification_module.readable_sample(previous_z).cpu()[None, :, :])

            sample_list_readable = torch.mean(torch.cat(sample_list_readable),0)
            sample_list = torch.mean(torch.cat(sample_list),0)

            if return_pred :
                return sample_list_readable, y_hat

            return sample_list_readable


