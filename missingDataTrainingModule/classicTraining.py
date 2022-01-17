from numpy.core.fromnumeric import reshape, var
from numpy.lib.npyio import _savez_compressed_dispatcher
from numpy.testing._private.utils import requires_memory
from psutil import test
from torch import neg, neg_
from torch._C import _show_config
from torch.nn.functional import batch_norm, mse_loss, one_hot
from .Selection import * 
from .Classification import *
from .Distribution import *
from .utils_missing import *
import time
import tqdm

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
        dic["accuracy_prediction_no_selection"] = correct.item()
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
            for batch_index, data in enumerate(loader.test_loader):
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

    def train_epoch(self, epoch, loader, nb_sample_z_monte_carlo = 10, nb_sample_z_IWAE = None, save_dic=False, verbose=False,):
        self.train()
        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)

            dic = self._train_step(data, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE=nb_sample_z_IWAE)
            
            if batch_idx % 100 == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            print(f"Learning Rate classification : {self.scheduler_classification.get_last_lr()}")
            self.scheduler_classification.step()
        
        return total_dic

        

    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 10, nb_sample_z_IWAE = None, ):
        
        
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        batch_size = data.size(0)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, 
                                                                                    target = target,
                                                                                    index=index,
                                                                                    one_hot_target = one_hot_target,
                                                                                    nb_sample_z_IWAE = None,
                                                                                    nb_sample_z_monte_carlo= nb_sample_z_monte_carlo,
                                                                                    nb_imputation = None)
        if index_expanded is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
        else :
            index_expanded_flatten = None

        p_z = self.fixed_distribution(data,)
        z = p_z.sample((nb_sample_z_monte_carlo, ))
        if self.reshape_mask_function is not None :
            z = self.reshape_mask_function(z)

        log_y_hat, _ = self.classification_module(data_expanded.flatten(0,1), z.detach(), index_expanded_flatten)
        log_y_hat = log_y_hat.reshape(nb_sample_z_monte_carlo * batch_size, dataset.get_dim_output())

        neg_likelihood = F.nll_loss(log_y_hat, target_expanded.flatten())
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target_expanded.flatten(0,1))**2,1))
        loss = neg_likelihood
        dic = self._create_dic(loss, neg_likelihood, mse_loss)
        loss.backward()
        self.optim_classification.step()
        return dic
    

        
    def _test_step(self, data, target, dataset, index, nb_sample_z = 1):
        batch_size = data.size(0)
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo= nb_sample_z, nb_imputation = None, nb_sample_z_IWAE= None)

        if index_expanded is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
        else :
            index_expanded_flatten = None

        self.fixed_distribution(data)
        z = self.fixed_distribution.sample((nb_sample_z, ))
        if self.reshape_mask_function is not None :
            z = self.reshape_mask_function(z)
        log_y_hat, _ = self.classification_module(data_expanded.flatten(0,1), z, index_expanded_flatten)
        log_y_hat = log_y_hat.reshape(nb_sample_z*batch_size, -1)

        neg_likelihood = F.nll_loss(log_y_hat, target_expanded.flatten())
        mse_current = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target_expanded.flatten(0,1))**2,1))
        return log_y_hat, neg_likelihood, mse_current

class GroundTruthSelectionTraining():
    def __init__(self, selection_module, reshape_mask_function) -> None:
        self.selection_module = selection_module
        self.reshape_mask_function = reshape_mask_function
        self.use_cuda = False

    def compile(self, optim_selection, scheduler_selection = None,):
        self.optim_selection = optim_selection
        self.scheduler_selection = scheduler_selection
        self.compiled = True

    def train(self):
        self.selection_module.train()

    def eval(self):
        self.selection_module.eval()
    def zero_grad(self):
        self.selection_module.zero_grad()

    def cuda(self):
        if not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
        else :
            self.selection_module.cuda()
            self.use_cuda = True
    
    def train_epoch(self, epoch, loader, verbose=False,):
        self.train()
        
        dataset = loader.dataset
        total_dic = {}
        for batch_idx, data in tqdm.tqdm(enumerate(loader.train_loader)):
            self.zero_grad()
            data, target, index = parse_batch(data)
            batch_size = data.shape[0]
            quadrant = dataset.quadrant_train[index]
            if self.use_cuda:
                data =data.cuda()
                target = target.cuda()
                quadrant = quadrant.cuda()
                index = index.cuda()
            log_pi_list, loss_reg = self.selection_module(data,)
            log_pi_list = self.reshape_mask_function(log_pi_list)
            quadrant = quadrant.reshape(log_pi_list.shape)

            log_pi_list = log_pi_list.flatten(1)
            quadrant = quadrant.flatten(1)
            mse = torch.mean(torch.sum((torch.exp(log_pi_list) - quadrant)**2, axis=-1))
            mse.backward()
            
            self.optim_selection.step()
            if batch_idx%100 == 0 :
                # for name, param in self.selection_module.named_parameters():
                    # print(name, param.grad)
                # fig, axs = plt.subplots(1,3, figsize=(15,5))
                # axs[0].imshow(data[0].reshape(28,56).detach().cpu().numpy(), cmap='gray')
                # axs[1].imshow(quadrant[0].reshape(28,56).detach().cpu().numpy(), cmap = 'gray')
                # axs[2].imshow(torch.exp(log_pi_list[0]).reshape(28,56).detach().cpu().numpy(), cmap='gray')
                # plt.show()
                print(
                    f"Epoch {epoch} Batch {batch_idx*batch_size/len(loader.train_loader.dataset)}/100 Loss {mse.item()}"
                )
            

        
        if self.scheduler_selection is not None :
            print(f"Learning Rate selection : {self.scheduler_selection.get_last_lr()}")
            self.scheduler_selection.step()
        
        return total_dic


    def test(self, epoch, loader, verbose = False,):
        mse = 0
        correct = 0

        self.eval()
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)
                batch_size = data.shape[0]
                quadrant = loader.dataset.quadrant_test[index]

                if self.use_cuda :
                    data, target, index = on_cuda(data, target = target, index = index,)
                    quadrant = quadrant.cuda()
                one_hot_target = get_one_hot(target, num_classes = loader.dataset.get_dim_output())


                log_pi_list, _ = self.selection_module(data)
                log_pi_list = self.reshape_mask_function(log_pi_list)
                quadrant = quadrant.reshape(log_pi_list.shape)
                mse += torch.sum(torch.pow(torch.exp(log_pi_list) - quadrant, 2)).detach().cpu().numpy()
                correct += torch.sum(torch.mean(torch.abs(torch.exp(log_pi_list)- quadrant), axis = -1)).detach().cpu().numpy()

            mse = mse / len(loader.test_loader.dataset)
            correct = correct / len(loader.test_loader.dataset)
            print('\nTest set: MSE: {:.4f}, Accuracy {:.4f}'.format(
                mse, correct, 
                ))
            print("\n")

        return None

    def _create_dic_test(self, correct, correct_no_selection, neg_likelihood, test_loss, pi_list_total, correct_post_hoc = None):
        total_dic = super()._create_dic_test(correct_no_selection, neg_likelihood, test_loss)
        treated_pi_list_total = np.concatenate(pi_list_total)
        total_dic["mean_pi_list"] = np.mean(treated_pi_list_total).item()
        total_dic["accuracy_prediction_selection"] = correct.item()
        q = np.quantile(treated_pi_list_total, [0.25,0.5,0.75])
        total_dic["pi_list_q1"] = q[0].item()
        total_dic["pi_list_median"] = q[1].item()
        total_dic["pi_list_q2"] = q[2].item()
        total_dic["correct_post_hoc"] = correct_post_hoc

        return total_dic



class SELECTION_BASED_CLASSIFICATION(ordinaryTraining):
    """ Abstract class to help the classification of the different module """

    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False, post_hoc = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False, ):

        super().__init__(classification_module,)
        self.selection_module = selection_module
        self.distribution_module = distribution_module
        self.baseline = baseline


        self.reshape_mask_function = reshape_mask_function
        self.show_variance_gradient = show_variance_gradient
              
        self.fix_classifier_parameters = fix_classifier_parameters
        self.post_hoc_guidance = post_hoc_guidance
        self.post_hoc = post_hoc
        self.argmax_post_hoc = argmax_post_hoc

        if self.fix_classifier_parameters :
            for param in self.classification_module.parameters():
                param.requires_grad = False

 
    def reshape(self, z,):
        reshaped_z = self.reshape_mask_function(z)
        return reshaped_z

    def calculate_variance(self, loss):
        if self.show_variance_gradient :
            g = []
            for k in range(len(loss)):
                g2 = vectorize_gradient(torch.autograd.grad(loss[k], self.selection_module.selector.parameters(), retain_graph = True, create_graph = False), 
                        self.selection_module.selector.named_parameters(), set_none_to_zero=True)
                g.append((g2).unsqueeze(1))
            g_mean = torch.mean(torch.cat(g, axis=1),axis=1)
            g_square = torch.mean(torch.cat(g,axis=1)**2,axis=1)
            variance = torch.mean(g_square - g_mean**2)
        else :
            variance = None
        return variance

    def compile(self, optim_classification, optim_selection, scheduler_classification = None, scheduler_selection = None, optim_baseline = None, scheduler_baseline = None, optim_distribution_module = None, scheduler_distribution_module = None, **kwargs):
        self.optim_classification = optim_classification
        self.scheduler_classification = scheduler_classification
        self.optim_selection = optim_selection
        self.scheduler_selection = scheduler_selection
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
        if not self.fix_classifier_parameters or self.optim_classification is None :
           self.optim_classification.step()
        self.optim_selection.step()
        if self.optim_distribution_module is not None :
            self.optim_distribution_module.step()

    
    def train_epoch(self, epoch, loader, nb_sample_z_monte_carlo = 3, nb_sample_z_IWAE = 3, save_dic=False, verbose=True):
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

    def _calculate_neg_likelihood(self, data, index, log_y_hat, target, one_hot_target):
        # Loss for classification
        if not self.post_hoc:
            neg_likelihood = F.nll_loss(log_y_hat, target.flatten(), reduce = False)
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target.reshape(log_y_hat.shape))**2,1)) 
        else :
            if self.post_hoc_guidance is not None :
                out_y, _ = self.post_hoc_guidance(data, index = index)
            elif self.fix_classifier_parameters :
                out_y, _ =self.classification_module(data, index=index)
            else :
                raise AttributeError("You can't have post-hoc without a post hoc guidance or fixing the classifier parameters")
            out_y = out_y.detach()
            mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-torch.exp(out_y).float())**2,1)) 
            if self.argmax_post_hoc :
                out_y = torch.argmax(out_y, -1)
                neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce = False)
            else :
                neg_likelihood = - torch.sum(torch.exp(out_y) * log_y_hat, -1)

        return neg_likelihood, mse_loss




    def _create_dic(self, loss_total, neg_likelihood, mse_loss, pi_list, loss_rec = None, loss_reg = None, loss_selection = None, variance_gradient = None):
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
        dic["loss_selection"] = get_item(loss_selection)
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
                data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z, nb_imputation = None)
                if index_expanded is not None :
                    index_expanded_flatten = index_expanded.flatten(0,1)
                else :
                    index_expanded_flatten = None



                log_pi_list, _ = self.selection_module(data)
                pi_list_total.append(torch.exp(log_pi_list).cpu().numpy())
                self.distribution_module(log_pi_list)
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
                log_y_hat_iwae = torch.logsumexp(log_y_hat_destructed,0) - torch.log(torch.tensor(nb_sample_z).type(torch.float32))
                log_y_hat_mean = torch.mean(log_y_hat_destructed, axis=0)
                index = torch.where(torch.any(torch.isnan(log_y_hat_destructed), axis=-1))[1]

                test_loss_likelihood += F.nll_loss(log_y_hat_destructed.flatten(0,1),target_expanded.flatten(0,1))
                test_loss_mse += torch.mean(torch.sum((torch.exp(log_y_hat_mean)-one_hot_target)**2,1))

                pred_destructed = torch.argmax(log_y_hat_mean, dim=1)
                correct_destructed += pred_destructed.eq(target).sum()



            test_loss_mse /= len(loader.test_loader.dataset) * batch_size
            print('\nTest set: MSE: {:.4f}, Likelihood {:.4f}, Accuracy No selection: {}/{} ({:.0f}%), Accuracy selection: {}/{} ({:.0f}%), Accuracy PostHoc: {}/{} ({:.0f}%),'.format(
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

    def _create_dic_test(self, correct, correct_no_selection, neg_likelihood, test_loss, pi_list_total, correct_post_hoc = None):
        total_dic = super()._create_dic_test(correct_no_selection, neg_likelihood, test_loss)
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
    def __init__(self, classification_module, selection_module, distribution_module, classification_distribution_module = FixedBernoulli(),
                baseline = None, reshape_mask_function = None, fix_classifier_parameters = False,
                post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,):
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters,
                        post_hoc = post_hoc, post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc,
                        show_variance_gradient = show_variance_gradient)
        if self.post_hoc_guidance is not None :
            raise NotImplementedError("REALX does not support post hoc guidance")
        self.classification_distribution_module = classification_distribution_module

        # if self.fix_classifier_parameters and self.post_hoc :
            # raise NotImplementedError("REALX does not support fix classifier")

    def _train_step(self, data, target, dataset, index = None,  nb_sample_z_monte_carlo = 1, nb_sample_z_IWAE = 1, need_dic = False):
        # start_time = time.time()
        self.zero_grad()

        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo= nb_sample_z_monte_carlo,  nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo= nb_sample_z_monte_carlo,  nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,1)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,1)
        else :
            index_expanded_flatten = None
        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,3)
            

        #### TRAINING CLASSIFICATION :
        if not self.fix_classifier_parameters :
            # Destructive module :
            log_pi_list, loss_reg = self.selection_module(data)
            log_pi_list.detach()

            # Train classification module :
            self.classification_distribution_module(log_pi_list)
            z = self.classification_distribution_module.sample((nb_sample_z_IWAE, nb_sample_z_monte_carlo,))
            z = self.reshape(z)
            
            # Classification module :
            log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,2), z, index_expanded_flatten)

            # Loss for classification:

            neg_likelihood, mse_loss = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                            index = index_expanded_multiple_imputation_flatten,
                                                            log_y_hat=log_y_hat,
                                                            target = target_expanded_multiple_imputation_flatten,
                                                            one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

            neg_likelihood = neg_likelihood.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size)
            neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32))
            neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32))
            neg_likelihood = torch.mean(neg_likelihood, axis=0)
            neg_likelihood = torch.sum(neg_likelihood)
            loss_classification_module = neg_likelihood
            loss_classification_module.backward()
            self.optim_classification.step()
            self.zero_grad()


        ### Train selection module :

        # Selection Module :
        log_pi_list, loss_reg = self.selection_module(data)
        

        # Distribution :
        try :
            self.distribution_module(log_pi_list)
        except :
            print(log_pi_list)

        sig_z, s, sig_z_tilde = self.distribution_module.sample((nb_sample_z_IWAE, nb_sample_z_monte_carlo))
        log_prob_pz = self.distribution_module.log_prob(s).reshape((nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, -1))
        log_prob_pz = torch.sum(torch.sum(log_prob_pz, axis = -1), axis=0)
        s = self.reshape(s)
        sig_z = self.reshape(sig_z)
        sig_z_tilde = self.reshape(sig_z_tilde)
       


        # Calculate
        # 1. f(s)
        f_s, _ = self.classification_module(data_expanded.flatten(0,2), s, index=index_expanded_flatten)
        # 2. c(z)
        c_z, _ = self.classification_module(data_expanded.flatten(0,2), sig_z , index=index_expanded_flatten)
        # 3. c(z~)
        c_z_tilde, _ = self.classification_module(data_expanded.flatten(0,2), sig_z_tilde, index=index_expanded_flatten)

        # Compute the probabilities 
        # 1. f(s)
        p_f_s, _ = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                index = index_expanded_multiple_imputation_flatten,
                                                log_y_hat = f_s,
                                                target = target_expanded_multiple_imputation_flatten,
                                                one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        p_f_s = p_f_s.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, )
        p_f_s = torch.logsumexp(p_f_s, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        p_f_s = torch.logsumexp(p_f_s, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log
                            
        # 2. c(z)
        p_c_z, _ = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                        index = index_expanded_multiple_imputation_flatten,
                                        log_y_hat = c_z,
                                        target = target_expanded_multiple_imputation_flatten,
                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)
        p_c_z = p_c_z.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, )
        p_c_z = torch.logsumexp(p_c_z, axis=0) - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        p_c_z = torch.logsumexp(p_c_z, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log


        # 3. c(z~)
        p_c_z_tilde, _ = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                        index = index_expanded_multiple_imputation_flatten,
                                        log_y_hat = c_z_tilde,
                                        target = target_expanded_multiple_imputation_flatten,
                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        
        p_c_z_tilde = p_c_z_tilde.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, )
        p_c_z_tilde = torch.logsumexp(p_c_z_tilde, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        p_c_z_tilde = torch.logsumexp(p_c_z_tilde, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log

        # Reward
        neg_reward = p_f_s - p_c_z_tilde
        neg_reward = neg_reward.detach()
        assert neg_reward.shape == log_prob_pz.shape
        neg_reward_prob = neg_reward * log_prob_pz



        # Terms to Make Expection Zero
        E_0 = p_c_z - p_c_z_tilde

        # Losses
        s_loss = neg_reward_prob + E_0
        s_loss = torch.mean(s_loss, axis=0)
        s_loss = torch.sum(s_loss)

        
        # for name, param in self.selection_module.named_parameters():
        #     print(name)
        #     grad_reinforce = torch.autograd.grad(neg_reinforce, param, retain_graph=True)[0]
        #     grad_sloss = torch.autograd.grad(s_loss, param, retain_graph=True)[0]
        #     print("REINFORCE", grad_reinforce[0])
        #     print("sLoss", grad_sloss[0])
        #     break
        # assert 1 == 0
        
        # Train
        loss_selection = s_loss + loss_reg



        # current_time = time.time()
        loss_selection.backward()
        self.optim_selection.step()
        self.optim_distribution_module.step()

        # end_time_aux = time.time()
        # print("Time for one backward : ", end_time_aux - current_time)

        if self.fix_classifier_parameters :
            if need_dic :
                dic = self._create_dic(loss_total = loss_selection+loss_reg,
                                    neg_likelihood= neg_likelihood,
                                    mse_loss= mse_loss,
                                    loss_rec = torch.tensor(0.0).type(torch.float32),
                                    loss_reg = loss_reg,
                                    loss_selection = s_loss,
                                    pi_list = torch.exp(log_pi_list))
            else :
                dic = {}
        else :
            if need_dic :
                dic = self._create_dic(loss_total = loss_selection+loss_classification_module+loss_reg,
                                    neg_likelihood= neg_likelihood,
                                    mse_loss= mse_loss,
                                    loss_rec = loss_classification_module,
                                    loss_reg = loss_reg,
                                    loss_selection = s_loss,
                                    pi_list = torch.exp(log_pi_list))
            else :
                dic = {}

        # end_time = time.time()
        # print("Time for one iteration : ", end_time - start_time)

        return dic

class ReparametrizedTraining(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, distribution_module,
                baseline = None, reshape_mask_function = None, fix_classifier_parameters = False, post_hoc = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,):

        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters,
                        post_hoc = post_hoc, post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc,
                        show_variance_gradient = show_variance_gradient)


    def _train_step(self, data, target, dataset, index = None,  nb_sample_z_monte_carlo = 3, nb_sample_z_IWAE = 3, need_dic = False):        
        self.zero_grad()
        
        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE = nb_sample_z_IWAE, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target,nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE = nb_sample_z_IWAE, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,2)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,3)
        else :
            index_expanded_flatten = None
            index_expanded_multiple_imputation_flatten = None
        
        # Destructive module :
        log_pi_list, loss_reg = self.selection_module(data)


        # Distribution :
        self.distribution_module(log_pi_list)
        z = self.distribution_module.sample((nb_sample_z_IWAE, nb_sample_z_monte_carlo, ))
        z = self.reshape(z)

        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,2), z, index_expanded_flatten)

        # Loss for classification:
        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,3)
        
        neg_likelihood, mse_loss = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                        index = index_expanded_multiple_imputation_flatten,
                                                        log_y_hat=log_y_hat,
                                                        target = target_expanded_multiple_imputation_flatten,
                                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        neg_likelihood = neg_likelihood.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size)


        # Loss for selection
        neg_likelihood = torch.logsumexp(neg_likelihood,0) - torch.log(torch.tensor(nb_imputation, dtype=torch.float32)) # Monte Carlo Estimator for the imputation
        neg_likelihood = torch.logsumexp(neg_likelihood,0) - torch.log(torch.tensor(nb_sample_z_IWAE, dtype=torch.float32)) # IWAE Loss for the selection
        neg_likelihood = torch.mean(neg_likelihood, 0) # Monte Carlo estimator for sampling z
        neg_likelihood = torch.sum(neg_likelihood) # Sum of the likelihood 


        # Updates 
        loss_rec = neg_likelihood
        loss_total = loss_rec + loss_reg + loss_reconstruction

        loss_total.backward()
        self.optim_step()

        # Measures :
        if need_dic :
            dic = self._create_dic(loss_total = loss_total,
                                neg_likelihood= neg_likelihood,
                                mse_loss= mse_loss,
                                loss_rec = loss_rec,
                                loss_reg = loss_reg,
                                loss_selection = None,
                                pi_list = torch.exp(log_pi_list))
        else :
            dic = {}

        return dic


        

class AllZTraining(SELECTION_BASED_CLASSIFICATION):
    """ Difference betzeen the previous is the way we calculate the multiplication with the loss"""
    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False, post_hoc = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,):
        
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters, post_hoc = post_hoc,
                        post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc, show_variance_gradient = show_variance_gradient)
        self.z = None
        self.computed_combination = False


    def _train_step(self, data, target, dataset, index = None,  nb_sample_z_monte_carlo = 1, nb_sample_z_IWAE = 1, need_dic = False):
        self.zero_grad()
        dim_total = np.prod(data.shape[1:])
        batch_size = data.shape[0]
        nb_sample_z_monte_carlo = 2**dim_total
        nb_imputation = self.classification_module.imputation.nb_imputation

        # Create all z combinations :
        if not self.computed_combination :
            self.z = get_all_z(dim_total)
            self.computed_combination = True
        z = self.z.unsqueeze(0).unsqueeze(-2)
        z = z.expand(nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, dim_total).detach()
        if self.use_cuda:
            z = z.cuda()
        
        # Prepare data :
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target,  nb_sample_z_IWAE = nb_sample_z_IWAE, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target,  nb_sample_z_IWAE=nb_sample_z_IWAE, nb_sample_z_monte_carlo=nb_sample_z_monte_carlo, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,2)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,3)
        
        
        # Destructive module :
        log_pi_list, loss_reg = self.selection_module(data)

        # Distribution :
        self.distribution_module(log_pi_list)  
        log_prob_pz = self.distribution_module.log_prob(z).reshape(nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, -1)
        log_prob_pz = torch.sum(torch.sum(log_prob_pz, axis = -1),axis=0) # Product of proba in dim and inside the IWAE
        z = self.reshape(z)
        
        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,2), z, index = index_expanded_flatten) # The expanding for multiple imputation is inside
    
        # Loss for classification:
        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,3)
        
        neg_likelihood, mse_loss = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                        index = index_expanded_multiple_imputation_flatten,
                                                        log_y_hat=log_y_hat,
                                                        target = target_expanded_multiple_imputation_flatten,
                                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        neg_likelihood = neg_likelihood.reshape(nb_imputation, nb_sample_z_IWAE,nb_sample_z_monte_carlo, batch_size)
        # Loss for selection :
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb_imputation
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # IWAE
        assert neg_likelihood.shape == log_prob_pz.shape
        neg_likelihood *= torch.exp(log_prob_pz) # MC sample z
        neg_likelihood = torch.sum(neg_likelihood, axis = 0) # Mc sample z
        neg_likelihood = torch.sum(neg_likelihood) # Batch size
        # neg_likelihood = torch.mean(neg_likelihood)

        # Update :
        loss_total = neg_likelihood + loss_reg
        loss_total.backward()
        self.optim_step()
        

        # Measures :
        if need_dic :
            dic = self._create_dic(loss_total,
                        neg_likelihood,
                        mse_loss,
                        loss_rec = loss_reconstruction, 
                        loss_reg = loss_reg, 
                        pi_list = torch.exp(log_pi_list))
        else :
            dic = {}
        return dic



class REINFORCE(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False, post_hoc = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,  ):
        
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters, post_hoc = post_hoc,
                        post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc, show_variance_gradient=show_variance_gradient,)



    def _train_step(self, data, target, dataset, index = None,  nb_sample_z_monte_carlo = 10, nb_sample_z_IWAE = 10, need_dic = False):

        self.zero_grad()
        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]

        # Get data :
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)

        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,2)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,3)
        else :
            index_expanded_flatten = None
            index_expanded_multiple_imputation_flatten = None
    
        # Prepare baseline :
        if self.baseline is not None:
            log_y_hat_baseline = self.baseline(data)
            # log_y_hat_baseline_masked = torch.masked_select(log_y_hat_baseline, one_hot_target>0.5)
            loss_baseline = F.nll_loss(log_y_hat_baseline, target, reduce = False) # Batch_size, 1

        # Selection Module :
        log_pi_list, loss_reg = self.selection_module(data)
        

        # Distribution :
        self.distribution_module(log_pi_list)
        z = self.distribution_module.sample((nb_sample_z_IWAE, nb_sample_z_monte_carlo))
        log_prob_pz = self.distribution_module.log_prob(z).reshape((nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, -1))
        log_prob_pz = torch.sum(torch.sum(log_prob_pz, axis = -1), axis=0)
        z = self.reshape(z)
        assert z.shape == data_expanded.flatten(0,2).shape

        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,2), z.detach(), index = index_expanded_flatten)
    


        # Choice for target :
        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,3)
        
        neg_likelihood, mse_loss = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                        index = index_expanded_multiple_imputation_flatten,
                                                        log_y_hat=log_y_hat,
                                                        target = target_expanded_multiple_imputation_flatten,
                                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        neg_likelihood = neg_likelihood.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size)

        # Loss for classification
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log
        neg_reward = neg_likelihood.clone().detach()
        neg_likelihood = torch.mean(neg_likelihood, axis=0) # MC estimator outside log
        neg_likelihood = torch.mean(neg_likelihood) # batch size
        loss_classification_module = neg_likelihood

        # Loss for selection :
        
        if self.baseline is not None :
            neg_reward = neg_reward - loss_baseline.detach().reshape(1, batch_size,).expand((nb_sample_z_monte_carlo, batch_size))

        assert log_prob_pz.shape == neg_reward.shape
        loss_hard = log_prob_pz*neg_reward # Log multiplication for REINFORCE
        loss_selection = torch.mean(loss_hard, axis = 0) # MCMC Estimator
        loss_selection = torch.sum(loss_selection) # Batch_size


        # Update :
        if self.baseline is not None :
            loss_total = loss_selection + loss_classification_module + loss_reg + loss_reconstruction + loss_baseline
        else :
            loss_total = loss_selection + loss_classification_module + loss_reg + loss_reconstruction
        
        loss_total.backward()
        self.optim_step()

        if need_dic :
            dic = self._create_dic(loss_total,
                    neg_likelihood,
                    mse_loss,
                    loss_rec = loss_reconstruction,
                    loss_reg = loss_reg,
                    pi_list = torch.exp(log_pi_list), 
                    loss_selection = loss_selection, 
                    variance_gradient = None)
        else :
            dic = {}
        return dic


## FOR THE MOMENT, NO LOOK AT THAT :


class REBAR(SELECTION_BASED_CLASSIFICATION):
    def __init__(self, classification_module, selection_module, distribution_module, baseline = None,
                reshape_mask_function = None, fix_classifier_parameters = False, post_hoc = False,
                post_hoc_guidance = None, argmax_post_hoc = False, show_variance_gradient = False,  ):
        
        super().__init__(classification_module, selection_module, distribution_module, baseline = baseline,
                        reshape_mask_function = reshape_mask_function, fix_classifier_parameters = fix_classifier_parameters, post_hoc=post_hoc,
                        post_hoc_guidance = post_hoc_guidance, argmax_post_hoc = argmax_post_hoc, show_variance_gradient=show_variance_gradient,)


    def set_variance_grad(self, network):
        grad = torch.nn.utils.parameters_to_vector(-p.grad for p in network.parameters()).flatten()
        variance = torch.mean(grad**2)
        return variance



    def _train_step(self, data, target, dataset, index = None,  nb_sample_z_monte_carlo = 10, nb_sample_z_IWAE = 10, need_dic = False):
        # start_time = time.time()
        self.zero_grad()
        nb_imputation = self.classification_module.imputation.nb_imputation
        batch_size = data.shape[0]
        # Get data :
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)

        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded, target_expanded, index_expanded, one_hot_target_expanded = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = None)
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = nb_imputation)
        if index is not None :
            index_expanded_flatten = index_expanded.flatten(0,2)
            index_expanded_multiple_imputation_flatten = index_expanded_multiple_imputation.flatten(0,3)
        else :
            index_expanded_flatten = None
            index_expanded_multiple_imputation_flatten = None
    

        ### TRAINING CLASSIFICATION :
        # Selection Module :
        log_pi_list, _ = self.selection_module(data)
        

        # Distribution :
        self.distribution_module.eval()
        self.distribution_module(log_pi_list)
        z = self.distribution_module.sample((nb_sample_z_IWAE, nb_sample_z_monte_carlo, ))
        z = self.reshape(z)


        

        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded.flatten(0,2), z.detach(), index = index_expanded_flatten)
    
        # Choice for target :

        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        one_hot_target_expanded_multiple_imputation_flatten = one_hot_target_expanded_multiple_imputation.flatten(0,3)
        
        neg_likelihood, mse_loss = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                        index = index_expanded_multiple_imputation_flatten,
                                                        log_y_hat=log_y_hat,
                                                        target = target_expanded_multiple_imputation_flatten,
                                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        neg_likelihood = neg_likelihood.reshape(nb_imputation, nb_sample_z_IWAE,nb_sample_z_monte_carlo, batch_size)
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log
        neg_likelihood = torch.mean(neg_likelihood, axis=0) # MC estimator outside log
        neg_likelihood = torch.sum(neg_likelihood) # batch size
        loss_classification_module = neg_likelihood
        loss_classification_module.backward()
        self.optim_classification.step()
        self.zero_grad()

        ### TRAINING SELECTION :

       # Selection Module :
        log_pi_list, loss_reg = self.selection_module(data)
        

        # Distribution :
        self.distribution_module.train()
        try :
            self.distribution_module(log_pi_list)
        except :
            print(log_pi_list)

        sig_z, s, sig_z_tilde = self.distribution_module.sample((nb_sample_z_IWAE, nb_sample_z_monte_carlo))
        log_prob_pz = self.distribution_module.log_prob(s).reshape((nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, -1))
        log_prob_pz = torch.sum(torch.sum(log_prob_pz, axis = -1), axis=0)
        s = self.reshape(s)
        sig_z = self.reshape(sig_z)
        sig_z_tilde = self.reshape(sig_z_tilde)
       


        # Calculate
        # 1. f(s)
        f_s, _ = self.classification_module(data_expanded.flatten(0,2), s, index=index_expanded_flatten)
        # 2. c(z)
        c_z, _ = self.classification_module(data_expanded.flatten(0,2), sig_z , index=index_expanded_flatten)
        # 3. c(z~)
        c_z_tilde, _ = self.classification_module(data_expanded.flatten(0,2), sig_z_tilde, index=index_expanded_flatten)

        # Compute the probabilities 
        # 1. f(s)
        p_f_s, _ = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                                index = index_expanded_multiple_imputation_flatten,
                                                log_y_hat = f_s,
                                                target = target_expanded_multiple_imputation_flatten,
                                                one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        p_f_s = p_f_s.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, )
        p_f_s = torch.logsumexp(p_f_s, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        p_f_s = torch.logsumexp(p_f_s, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log
                            
        # 2. c(z)
        p_c_z, _ = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                        index = index_expanded_multiple_imputation_flatten,
                                        log_y_hat = c_z,
                                        target = target_expanded_multiple_imputation_flatten,
                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)
        p_c_z = p_c_z.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, )
        p_c_z = torch.logsumexp(p_c_z, axis=0) - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        p_c_z = torch.logsumexp(p_c_z, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log


        # 3. c(z~)
        p_c_z_tilde, _ = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                                        index = index_expanded_multiple_imputation_flatten,
                                        log_y_hat = c_z_tilde,
                                        target = target_expanded_multiple_imputation_flatten,
                                        one_hot_target = one_hot_target_expanded_multiple_imputation_flatten)

        
        p_c_z_tilde = p_c_z_tilde.reshape(nb_imputation, nb_sample_z_IWAE, nb_sample_z_monte_carlo, batch_size, )
        p_c_z_tilde = torch.logsumexp(p_c_z_tilde, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32)) # Nb imputation inside log
        p_c_z_tilde = torch.logsumexp(p_c_z_tilde, axis=0) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32)) # Iwae loss inside log

        # Reward
        neg_reward = p_f_s - p_c_z_tilde
        neg_reward = neg_reward.detach()
        assert neg_reward.shape == log_prob_pz.shape
        neg_reward_prob = neg_reward * log_prob_pz

        # neg_reinforce = torch.mean(p_f_s.detach() * log_prob_pz, axis=0)
        # neg_reinforce = torch.sum(neg_reinforce)

        # Terms to Make Expection Zero
        E_0 = p_c_z - p_c_z_tilde

        # Losses
        s_loss = neg_reward_prob + E_0
        s_loss = torch.mean(s_loss, axis=0)
        s_loss = torch.sum(s_loss)

        
        # for name, param in self.selection_module.named_parameters():
        #     print(name)
        #     grad_reinforce = torch.autograd.grad(neg_reinforce, param, retain_graph=True)[0]
        #     grad_sloss = torch.autograd.grad(s_loss, param, retain_graph=True)[0]
        #     print("REINFORCE", grad_reinforce[0])
        #     print("sLoss", grad_sloss[0])
        #     break

        # assert 1==0

        # Train
        loss_selection = s_loss + loss_reg
        loss_selection.backward()

        self.optim_selection.step()
        self.optim_distribution_module.step()

        if need_dic :
            dic = self._create_dic(loss_total = loss_selection+loss_classification_module+loss_reg,
                                neg_likelihood= neg_likelihood,
                                mse_loss= mse_loss,
                                loss_rec = loss_classification_module,
                                loss_reg = loss_reg,
                                loss_selection = s_loss,
                                pi_list = torch.exp(log_pi_list))
        else :
            dic = {}

        # end_time = time.time()
        # print("Time per step : ", (end_time - start_time))

        return dic
