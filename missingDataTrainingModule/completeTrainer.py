from .Destruction import * 
from .Classification import *
from .utils_missing import *
import numpy as np

import matplotlib.pyplot as plt
def prepare_data(data, target, num_classes=10, use_cuda = False):
    if use_cuda:
        data =data.cuda()
        target = target.cuda()
    one_hot_target = torch.nn.functional.one_hot(target, num_classes = num_classes)

    return data, target, one_hot_target
    
def prepare_data_augmented(data, target, num_classes=10, Nexpectation = 1, use_cuda = False):
    if use_cuda:
        data =data.cuda()

        if target is not None :
            target = target.cuda()
    if target is not None :
        one_hot_target = torch.nn.functional.one_hot(target,num_classes=num_classes) # batch_size, category
        one_hot_target_expanded = one_hot_target.unsqueeze(0).expand(Nexpectation,-1,-1) #N_expectations, batch_size, category
    else :
        one_hot_target = None
        one_hot_target_expanded = None
    
    shape = data.shape
    data_unsqueezed = data.unsqueeze(0)


    wanted_transform = tuple(np.insert(-np.ones(len(shape),dtype = int),0,Nexpectation))
    data_expanded = data_unsqueezed.expand(wanted_transform) # N_expectation, batch_size, channels, size:...
    data_expanded_flatten = data_expanded.flatten(0,1)

    wanted_shape_flatten = data_expanded_flatten.shape

    return data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten

def print_dic(epoch, batch_idx, dic, dataset):
    percent = 100. * batch_idx / len(dataset.train_loader)
    to_print = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(epoch, batch_idx * dataset.batch_size_train, len(dataset.train_loader.dataset), percent)
    for key in dic.keys():
        to_print += "{}: {:.5f} \t".format(key, dic[key])
    print(to_print)

def save_dic_helper(total_dic, dic):
    if len(total_dic.keys())==0:
        for element in dic.keys():
            total_dic[element] = [dic[element]]
    else:
        for element in dic.keys():
            total_dic[element].append(dic[element])

    return total_dic

class ordinaryTraining():
    def __init__(self, classification_module, use_cuda = True, feature_extractor = None, kernel_patch = (1,1), stride_patch=(1,1)):
        if use_cuda == True and not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
            self.use_cuda = False
        else :
            self.use_cuda = use_cuda

        self.kernel_patch= kernel_patch
        self.stride_patch = stride_patch        
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None :
            self.need_feature = False
        else : 
            self.need_feature = True

    
        self.classification_module = classification_module
        self.classification_module.kernel_update(self.kernel_patch, self.stride_patch)
        if self.use_cuda :
            self.classification_module.cuda()
            if self.need_feature :
                self.feature_extractor.cuda()

      
    def _create_dic(self,loss, neg_likelihood, mse_loss):
        dic = {}
        dic["likelihood"] = -neg_likelihood.item()
        dic["mse_loss"] = mse_loss.item()
        dic["total_loss"] = loss.item()
        return dic

    def _create_dic_test(self, correct, neg_likelihood, mse_loss):
        dic = {}
        dic["correct"] = correct.item()
        dic["likelihood"] = -neg_likelihood.item()
        dic["mse"] = mse_loss.item()

        return dic


    def parameters(self):
        return self.classification_module.parameters()

    def zero_grad(self):
        self.classification_module.zero_grad()
        if self.need_feature :
            self.feature_extractor.zero_grad()

    def _predict(self, data):
        log_y_hat, loss_reconstruction = self.classification_module(data)
        return log_y_hat, loss_reconstruction

    def _train_step(self, data, target, dataset):
        self.zero_grad()

        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
        log_y_hat, _ = self._predict(data)

        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        loss = neg_likelihood
        dic = self._create_dic(loss, neg_likelihood, mse_loss)
        loss.backward()
        return dic


    def train(self, epoch, dataset, optim_classifier, optim_feature_extractor= None, save_dic = False):
        
        total_dic = {}
        self.classification_module.train()

        if self.need_feature :
            if optim_feature_extractor is None :
                raise AssertionError("Optimisation for feature extractor is needed")
            else :
                self.feature_extractor.train()

        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            dic = self._train_step(data,target,dataset)
            optim_classifier.step()
            if self.need_feature :
                optim_feature_extractor.step()

            if batch_idx % 100 == 0:
                print_dic(epoch, batch_idx, dic, dataset)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

        return total_dic



    def _test_step(self, data, target):
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
        log_y_hat, _ = self.classification_module(data)
        
        
        neg_likelihood = F.nll_loss(log_y_hat, target)
        mse_current = torch.mean(torch.sum((torch.exp(log_y_hat)-one_hot_target)**2,1))
        correct_current = pred.eq(target.data.view_as(pred)).sum()

        return log_y_hat, neg_likelihood, mse_current, correct_current


    def test(self,dataset):
        self.classification_module.eval()
        if self.need_feature :
            self.feature_extractor.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataset.test_loader:
                log_y_hat, neg_likelihood, mse_current, correct_current = self._test_step(data, target)
                test_loss += mse_current
                pred = log_y_hat.data.max(1, keepdim=True)[1]
                correct += correct_current


        test_loss /= len(dataset.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataset.test_loader.dataset),
        100. * correct / len(dataset.test_loader.dataset)))

        total_dic = self._create_dic_test(correct/len(dataset.test_loader.dataset), neg_likelihood, test_loss)

        return total_dic




    

class noVariationalTraining(ordinaryTraining):
    def __init__(self, classification_module, destruction_module, feature_extractor = None, use_cuda = True, kernel_patch = (1,1), stride_patch = (1,1)):
        super().__init__(classification_module, use_cuda = use_cuda, feature_extractor=feature_extractor, kernel_patch= kernel_patch, stride_patch = stride_patch)
        self.destruction_module = destruction_module
        
        self.destruction_module.kernel_update(kernel_patch, stride_patch)
        if self.use_cuda :
            self.destruction_module.cuda()
            if self.need_feature :
                self.feature_extractor.cuda()


    def zero_grad(self):
        self.classification_module.zero_grad()
        self.destruction_module.zero_grad()
        if self.need_feature :
            self.feature_extractor.zero_grad()

    def eval(self):
        self.classification_module.eval()
        self.destruction_module.eval()
        if self.need_feature :
            self.feature_extractor.eval()



    def train(self):
        self.classification_module.train()
        self.destruction_module.train()
        if self.need_feature :
            self.feature_extractor.train()
        
        


    def _create_dic(self, loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list):
        dic = super()._create_dic(loss_total, neg_likelihood, mse_loss)
        dic["loss_rec"] = loss_rec.item()
        dic["loss_reg"] = loss_reg.item()
        dic["mean_pi"] = torch.mean(torch.mean(pi_list.flatten(1),1)).item()
        quantiles = torch.tensor([0.25,0.5,0.75])
        if self.use_cuda: 
            quantiles = quantiles.cuda()
        q = torch.quantile(pi_list.flatten(1),quantiles,dim=1,keepdim = True)
        dic["median_pi"] = torch.mean(q[1]).item()
        dic["q1_pi"] = torch.mean(q[0]).item()
        dic["q2_pi"] = torch.mean(q[2]).item()
        if self.classification_module.imputation.has_constant():
            if torch.is_tensor(self.classification_module.imputation.get_constant()):
                dic["constantLeanarble"]= self.classification_module.imputation.get_constant().item()
        return dic

    def _get_pi(self, data):
        pi_list, loss_reg = self.destruction_module(data)
        return pi_list, loss_reg


    
    def _sample_z_test(self, pi_list, sampling_distribution, Nexpectation):
        p_z = sampling_distribution(pi_list)
        z = p_z.sample((Nexpectation,))
        
        return z, p_z

    def _destructive_test(self, data, sampling_distribution, Nexpectation):
        pi_list, loss_reg = self._get_pi(data)
        z, p_z = self._sample_z_test(pi_list, sampling_distribution, Nexpectation)
        return pi_list, loss_reg, z, p_z

    def _sample_z_train(self, pi_list, sampling_distribution, Nexpectation):
        p_z = sampling_distribution(pi_list)
        z = p_z.rsample((Nexpectation,))
        return z, p_z
    
    def _destructive_train(self, data, sampling_distribution, Nexpectation):
        pi_list, loss_reg = self._get_pi(data)
        z, p_z = self._sample_z_train(pi_list, sampling_distribution, Nexpectation)
        return pi_list, loss_reg, z, p_z

    def _predict(self, data, sampling_distribution, dataset, Nexpectation = 10, complete_output = False):
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten = prepare_data_augmented(data, None, num_classes=dataset.get_category(), Nexpectation=Nexpectation, use_cuda=self.use_cuda)

        pi_list, loss_reg, z, p_z = self._destructive_test(data, sampling_distribution, Nexpectation)

        y_hat, _ = self.classification_module(data_expanded_flatten, z)
        y_hat = y_hat.reshape(Nexpectation, -1, dataset.get_category())
        y_hat_mean = torch.logsumexp(y_hat,0)

        return y_hat_mean


        

    def _train_step(self, data, target, dataset, optim_classifier, optim_destruction, sampling_distribution, optim_feature_extractor= None, lambda_reg = 0.0, Nexpectation = 10, lambda_reconstruction = 0.0):
        self.zero_grad()
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten = prepare_data_augmented(data, target, num_classes=dataset.get_category(), Nexpectation=Nexpectation, use_cuda=self.use_cuda)
        
        # Destructive module :
        pi_list, loss_reg, z, p_z = self._destructive_train(data, sampling_distribution, Nexpectation)

        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z)
        log_y_hat = log_y_hat.reshape(Nexpectation, -1, dataset.get_category())
        log_y_hat_iwae = torch.logsumexp(log_y_hat,0) + torch.log(torch.tensor(1./Nexpectation))


        # Loss for regularization :
        loss_reconstruction = lambda_reconstruction * loss_reconstruction
        loss_reg = lambda_reg * loss_reg
        
        # Loss for classification
        neg_likelihood = F.nll_loss(log_y_hat_iwae, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat_iwae)-one_hot_target.float())**2,1)) 


        # Updates 
        loss_rec = neg_likelihood
        loss_total = loss_rec + loss_reg + loss_reconstruction
        loss_total.backward()

        optim_classifier.step()
        optim_destruction.step()
        if self.need_feature :
            optim_feature_extractor.step()

        dic = self._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list)

        return dic


    def train(self,epoch, dataset, optim_classifier, optim_destruction,  sampling_distribution, optim_feature_extractor=None, lambda_reg = 0.0, Nexpectation=10, save_dic = False, lambda_reconstruction = 0.0):
        self.classification_module.train()
        self.destruction_module.train()
        if self.need_feature :
            if optim_feature_extractor is None :
                raise AssertionError("Optimisation for feature extractor is needed")
            else :
                self.feature_extractor.train()

        total_dic = {}

        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            dic = self._train_step(data, target,dataset, optim_classifier, optim_destruction, sampling_distribution, optim_feature_extractor = optim_feature_extractor, lambda_reg = lambda_reg, Nexpectation = Nexpectation, lambda_reconstruction= lambda_reconstruction)


            if batch_idx % 100 == 0:
                print_dic(epoch, batch_idx, dic, dataset)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

        return total_dic
        
    def _create_dic_test(self, correct, neg_likelihood, test_loss, pi_list_total):
        total_dic = super()._create_dic_test(correct, neg_likelihood, test_loss)
        treated_pi_list_total = np.concatenate(pi_list_total)
        total_dic["mean_pi_list"] = np.mean(treated_pi_list_total).item()
        q = np.quantile(treated_pi_list_total, [0.25,0.5,0.75])
        total_dic["pi_list_q1"] = q[0].item()
        total_dic["pi_list_median"] = q[1].item()
        total_dic["pi_list_q2"] = q[2].item()

        return total_dic


    def test_no_var(self, dataset, sampling_distribution, Nexpectation = 10):
        test_loss_mse = 0
        test_loss_likelihood = 0
        correct = 0
        self.classification_module.eval()
        self.destruction_module.eval()
        if self.need_feature :
            self.feature_extractor.eval()


        pi_list_total = []
        with torch.no_grad():
            
            for data, target in dataset.test_loader:
                batch_size = data.shape[0]

                data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda, Nexpectation=Nexpectation)
                
                pi_list, _, z, p_z = self._destructive_test(data, sampling_distribution, Nexpectation)
                pi_list_total.append(pi_list.cpu().numpy())

                log_y_hat, _ = self.classification_module(data_expanded_flatten, z)
                log_y_hat = log_y_hat.reshape(Nexpectation, -1, dataset.get_category())
                log_y_hat_iwae = torch.logsumexp(log_y_hat,0)


                test_loss_likelihood += F.nll_loss(log_y_hat_iwae,target)
                test_loss_mse += torch.mean(torch.sum((torch.exp(log_y_hat_iwae)-one_hot_target)**2,1))

                pred = torch.argmax(log_y_hat_iwae,dim = 1)
                correct += pred.eq(target).sum()


            test_loss_mse /= len(dataset.test_loader.dataset) * batch_size
            print('\nTest set: MSE: {:.4f}, Likelihood {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                -test_loss_likelihood.item(), test_loss_mse.item(), correct.item(), len(dataset.test_loader.dataset),
                100. * correct.item() / len(dataset.test_loader.dataset)))
            print("\n")
            total_dic = self._create_dic_test(correct/len(dataset.test_loader.dataset),
                test_loss_likelihood,
                test_loss_mse,
                pi_list_total)

        return total_dic



    def MCMC(self, dataset, data, target, sampling_distribution, Niter, Nexpectation = 1,  eps = 1e-6, burn = 1000, jump = 50, return_pred = False):
        self.classification_module.eval()
        self.destruction_module.eval()
        if self.need_feature :
            self.feature_extractor.eval()


        with torch.no_grad():
            sample_list = []
            sample_list_readable = []
            y_hat_list = []
            data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda, Nexpectation=Nexpectation)


            pi_list, _, previous_z, pz = self._destructive_test(data, sampling_distribution, Nexpectation)
            log_y_hat, _ = self.classification_module(data_expanded_flatten, previous_z)
            log_y_hat_iwae = torch.logsumexp(log_y_hat, 0)
            previous_log_py = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)




            for k in range(Niter):
                z, p_z = self._sample_z_test(pi_list, sampling_distribution, Nexpectation)

                log_y_hat, _  = self.classification_module(data_expanded_flatten, z)
                log_y_hat = log_y_hat.reshape(Nexpectation, -1, dataset.get_category())
                log_y_hat_iwae = torch.logsumexp(log_y_hat, 0)

                log_py = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)


                u = torch.rand((log_py.shape)) 
                if self.use_cuda:
                    u = u.cuda()

                proba_acceptance = torch.exp(log_py-previous_log_py)
                mask_acceptance = u<proba_acceptance
                mask_acceptance = mask_acceptance.unsqueeze(1).expand((-1,z.shape[-1]))
                previous_z = torch.where(mask_acceptance, z, previous_z)
                log_y_hat, _  = self.classification_module(data_expanded_flatten, previous_z)
                log_y_hat = log_y_hat.reshape(Nexpectation, -1, dataset.get_category())
                log_y_hat_iwae = torch.logsumexp(log_y_hat, 0)
                previous_log_py = torch.masked_select(log_y_hat_iwae, one_hot_target>0.5)

                if k > burn and k%jump == 0 :
                    sample_list.append(previous_z.cpu()[None,:,:])
                    sample_list_readable.append(self.classification_module.imputation.readable_sample(previous_z).cpu()[None, :, :])

            sample_list_readable = torch.mean(torch.cat(sample_list_readable),0)
            sample_list = torch.mean(torch.cat(sample_list),0)

            if return_pred :
                return sample_list_readable, log_y_hat_iwae

            return sample_list_readable


class postHocTraining(noVariationalTraining):
    def __init__(self, classification_module, destruction_module,feature_extractor = None, kernel_patch = (1,1), stride_patch = (1,1), feature_extractor_training = False):
        super().__init__(classification_module, destruction_module, feature_extractor= feature_extractor, kernel_patch = kernel_patch, stride_patch = stride_patch)
        self.feature_extractor_training = feature_extractor_training
        self.classification_module.fix_parameters()
        if self.need_feature and feature_extractor_training :
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

class noVariationalTraining_REINFORCE(noVariationalTraining):
    def __init__(self, classification_module, destruction_module, critic = None, feature_extractor = None, kernel_patch = (1,1), stride_patch = (1,1), feature_extractor_training = False):
        super().__init__(classification_module, destruction_module, feature_extractor= feature_extractor, kernel_patch = kernel_patch, stride_patch = stride_patch)

        self.critic = critic
        if self.use_cuda and self.critic is not None :
            self.critic = self.critic.cuda()
        self.optim_critic = None

    def set_optim_critic(self, optim_critic):
        self.optim_critic = optim_critic

    def _destructive_train(self, data, sampling_distribution, Nexpectation):
        pi_list, loss_reg = self._get_pi(data)
        z, p_z = self._sample_z_test(pi_list, sampling_distribution, Nexpectation)
        return pi_list, loss_reg, z, p_z

    def _train_step(self, data, target, dataset, optim_classifier, optim_destruction, sampling_distribution, optim_feature_extractor =None, lambda_reg = 0.0, Nexpectation = 10, lambda_reconstruction = 0.0):
        self.zero_grad()
        if self.critic is not None :
            self.critic.zero_grad()
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten = prepare_data_augmented(data, target, num_classes=dataset.get_category(), Nexpectation=Nexpectation, use_cuda=self.use_cuda)
        
        # Destructive module :
        pi_list, loss_reg, z, p_z = self._destructive_train(data, sampling_distribution, Nexpectation)
        
        
        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z)
        log_y_hat = log_y_hat.reshape(Nexpectation, -1, dataset.get_category())
        log_y_hat_iwae = torch.logsumexp(log_y_hat,0) + torch.log(torch.tensor(1./Nexpectation))

      



        # Loss for regularization :
        loss_reconstruction = lambda_reconstruction * loss_reconstruction

        
        # Loss for classification
        neg_likelihood = F.nll_loss(log_y_hat_iwae, target)
        mse_loss = torch.mean(torch.sum((torch.exp(log_y_hat_iwae)-one_hot_target.float())**2,1)) 


        # Updates on classifier
        loss_rec = neg_likelihood

        loss_total = loss_rec + loss_reconstruction 


        loss_total.backward()
        optim_classifier.step()


        if self.optim_critic is not None and self.critic is not None:
            self.zero_grad()
            self.critic.zero_grad()
            log_y_hat_critic = self.critic(data)
            neg_likelihood_critic = F.nll_loss(log_y_hat_critic, target)
            neg_likelihood_critic.backward()
            self.optim_critic.step()


        # Updates on destruction :
        self.zero_grad()
        if self.critic is not None :
            self.critic.zero_grad()


        # Destructive module :
        pi_list, loss_reg = self._get_pi(data)
        loss_reg = lambda_reg * loss_reg
        log_prob_sampled_z = torch.sum(p_z.log_prob(z),-1).reshape(Nexpectation, -1)
        
        
        # Classification module :
        log_y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z)
        log_y_hat = log_y_hat.reshape(Nexpectation, -1, dataset.get_category())
        if self.critic is not None :
            log_y_hat_critic = self.critic(data_expanded_flatten)
            log_y_hat_critic = log_y_hat_critic.reshape(Nexpectation, -1, dataset.get_category())
            log_y_hat = (log_y_hat - log_y_hat_critic)
        log_y_hat_selected = torch.masked_select(log_y_hat, one_hot_target_expanded>0.5).reshape(Nexpectation, -1)


            

        log_y_hat_pz = log_y_hat_selected * log_prob_sampled_z

        loss_destructor = - torch.sum(torch.logsumexp(log_y_hat_pz,0))
        loss_total = loss_destructor + loss_reg
        loss_total.backward()
        optim_destruction.step()

        dic = self._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list)

        return dic


        


class variationalTraining(noVariationalTraining):
    def __init__(self, classification_module, destruction_module, destruction_module_var, feature_extractor = None, kernel_patch = (1,1), stride_patch = (1,1)):
        super().__init__(classification_module, destruction_module, feature_extractor, kernel_patch, stride_patch)
        self.destruction_module_var = destruction_module_var

        self.destruction_module_var.kernel_update(kernel_patch, stride_patch)
        if self.use_cuda :
            self.destruction_module_var.cuda()

    

    def _prob_calc(self, y_hat, one_hot_target_expanded, z , pz, qz):
        Nexpectation = one_hot_target_expanded.shape[0]
        log_prob_y = torch.masked_select(y_hat,one_hot_target_expanded>0.5).reshape(Nexpectation,-1)
        log_prob_pz = torch.sum(pz.log_prob(z),-1)
        log_prob_qz = torch.sum(qz.log_prob(z),-1)
        
        return log_prob_y,log_prob_pz,log_prob_qz
    
    def _likelihood_var(self, y_hat, one_hot_target_expanded, z, pz, qz):
        log_prob_y, log_prob_pz, log_prob_qz = self._prob_calc(y_hat, one_hot_target_expanded, z, pz, qz)
        return torch.mean(torch.logsumexp(log_prob_y+log_prob_pz-log_prob_qz,0))


    def _train_step(self, data, target, dataset, sampling_distribution, sampling_distribution_var, lambda_reg = 0.0, lambda_reg_var= 0.0, Nexpectation = 10, lambda_reconstruction = 0.0):
        self.classification_module.zero_grad()
        self.destruction_module.zero_grad()
        self.destruction_module_var.zero_grad()
        if self.need_feature :
            self.feature_extractor.cuda()


        batch_size = data.shape[0]
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), Nexpectation=Nexpectation, use_cuda=self.use_cuda)

        pi_list, loss_reg, _, _ = self.destruction_module(data)
        pi_list_var, loss_reg_var, _, _ = self.destruction_module_var(data, one_hot_target = one_hot_target)
        loss_reg = lambda_reg * loss_reg
        loss_reg_var = lambda_reg_var * loss_reg_var

        pz = sampling_distribution(pi_list)
        qz = sampling_distribution_var(pi_list_var)
        
        
        z = qz.rsample((Nexpectation,))
        
        y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z)
        y_hat = y_hat.reshape((Nexpectation,batch_size, dataset.get_category()))
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

    def train(self, epoch, dataset, optim_classifier, optim_destruction, optim_destruction_var, sampling_distribution, sampling_distribution_var, optim_feature_extractor = None, lambda_reg = 0.0, lambda_reg_var= 0.0, Nexpectation = 10, lambda_reconstruction = 0.0, save_dic = False):
        self.classification_module.train()
        self.destruction_module.train()
        self.destruction_module_var.train()
        total_dic = {}
        for batch_idx, (data, target) in enumerate(dataset.train_loader):

            dic = self._train_step(
                data, target, dataset,
                sampling_distribution, sampling_distribution_var,
                lambda_reg = lambda_reg, lambda_reg_var= lambda_reg_var,
                Nexpectation = Nexpectation, lambda_reconstruction=lambda_reconstruction
                )
                    
            if batch_idx % 100 == 0:
                print_dic(epoch, batch_idx, dic, dataset)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

            optim_classifier.step()
            optim_destruction.step()
            optim_destruction_var.step()

        return total_dic


 

            
    def test_var(self, dataset, sampling_distribution, sampling_distribution_var, Nexpectation = 10):
        self.classification_module.eval()
        self.destruction_module.eval() 
        self.destruction_module_var.eval()


        test_loss_mse = 0
        test_loss_likelihood = 0
        correct = 0
        
        
        with torch.no_grad():
            pi_list_total = []
            pi_list_var_total = []
            for data, target in dataset.test_loader:
                batch_size = data.shape[0]
                data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)

                pi_list, _, _, _ = self.destruction_module(data, test=True)
                pi_list_var, _, _, _ = self.destruction_module_var(data, one_hot_target=one_hot_target, test=True)
                pi_list_total.append(pi_list.cpu().numpy())
                pi_list_var_total.append(pi_list_var.cpu().numpy())

                pz = sampling_distribution(pi_list)
                qz = sampling_distribution_var(pi_list_var)

                z = qz.sample()

                y_hat, loss_reconstruction = self.classification_module(data_expanded_flatten, z)
                y_hat_squeeze = y_hat.squeeze()

                test_loss_likelihood = self._likelihood_var(y_hat,one_hot_target_expanded,z,pz,qz)
                test_loss_mse += torch.sum(torch.sum((torch.exp(y_hat_squeeze)-one_hot_target)**2,1))
                # pred = y_hat_squeeze.data.max(1, keepdim=True)[1]
                # correct += pred.eq(target.data.view_as(pred)).sum()
                pred = torch.argmax(y_hat,dim = 1)
                correct += pred.eq(target).sum()

            test_loss_mse /= len(dataset.test_loader.dataset) * dataset.batch_size_test
            print('\nTest set: AMSE: {:.4f}, Likelihood {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss_likelihood, test_loss_mse, correct, len(dataset.test_loader.dataset),
                100. * correct / len(dataset.test_loader.dataset)))
            print("\n")
            total_dic = self._create_dic_test(correct/len(dataset.test_loader.dataset),
                test_loss_likelihood,
                test_loss_mse,
                pi_list_total,
                pi_list_var_total)

            return total_dic




    def MCMC_var(self, dataset, data, target, sampling_distribution, sampling_distribution_var, Niter, eps = 1e-6, burn = 1000, jump = 50, return_pred = False):
        self.classification_module.eval()
        self.destruction_module.eval()
        with torch.no_grad():
            sample_list = []
            sample_list_readable = []

            data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
            batch_size = data.shape[0]
            input_size = data.shape[1] * data.shape[2] * data.shape[3]

            pi_list, _, _, _ = self.destruction_module(data, test=True)
            pi_list_var, _, _, _ = self.destruction_module_var(data, one_hot_target=one_hot_target, test=True)
            pz = sampling_distribution(pi_list)
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
                    sample_list_readable.append(self.classification_module.imputation.readable_sample(previous_z).cpu()[None, :, :])

            sample_list_readable = torch.mean(torch.cat(sample_list_readable),0)
            sample_list = torch.mean(torch.cat(sample_list),0)

            if return_pred :
                # print(y_hat_list)
                # print(torch.cat(y_hat_list))
                # y_hat_list = torch.mean(torch.cat(y_hat_list),0)
            
                return sample_list_readable, y_hat

            return sample_list_readable