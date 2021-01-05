from .Destruction import * 
from .Classification import *
from .utils_missing import *
import numpy as np

def prepare_data(data, target, num_classes=10, use_cuda = False):
    if use_cuda:
        data =data.cuda()
        target = target.cuda()
    one_hot_target = torch.nn.functional.one_hot(target, num_classes = num_classes)

    return data, target, one_hot_target
    
def prepare_data_augmented(data, target, num_classes=10, Nexpectation = 1, use_cuda = False):
    if use_cuda:
        data =data.cuda()
        target = target.cuda()
    
    one_hot_target = torch.nn.functional.one_hot(target,num_classes=num_classes) # batch_size, category
    one_hot_target_expanded = one_hot_target.unsqueeze(0).expand(Nexpectation,-1,-1) #N_expectations, batch_size, category
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


class ordinaryTraining():
    def __init__(self, classification_module, use_cuda = True, kernel_patch = (1,1), stride_patch=(1,1)):
        if use_cuda == True and not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
            self.use_cuda = False
        else :
            self.use_cuda = use_cuda

        self.kernel_patch= kernel_patch
        self.stride_patch = stride_patch        
        self.classification_module = classification_module
        self.classification_module.kernel_update(self.kernel_patch, self.stride_patch)
        if self.use_cuda :
            self.classification_module.cuda()

    def parameters(self):
        return self.classification_module.parameters()

    def _train_step(self, data, target, dataset):
        self.classification_module.zero_grad()
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
        y_hat = self.classification_module(data).squeeze()

        neg_likelihood = F.nll_loss(y_hat, target)
        mse_loss = torch.mean(torch.sum((torch.exp(y_hat)-one_hot_target)**2,1))
        loss = neg_likelihood

        dic = self._create_dic(loss, neg_likelihood, mse_loss)
        loss.backward()
        return dic
  
    def _create_dic(self,loss, neg_likelihood, mse_loss):
        dic = {}
        dic["likelihood"] = -neg_likelihood.item()
        dic["mse_loss"] = mse_loss.item()
        dic["total_loss"] = loss.item()
        return dic




    def train(self,epoch, dataset, optim_classifier):

        self.classification_module.train()
        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            dic = self._train_step(data,target,dataset)
            optim_classifier.step()

            if batch_idx % 100 == 0:
                print_dic(epoch, batch_idx, dic, dataset)



        

    def test(self,dataset):
        self.classification_module.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataset.test_loader:
                data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)

                y_hat = self.classification_module(data).squeeze()
                
                test_loss += torch.mean(torch.sum((y_hat-one_hot_target)**2,1))
                pred = y_hat.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(dataset.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataset.test_loader.dataset),
        100. * correct / len(dataset.test_loader.dataset)))


    

class noVariationalTraining(ordinaryTraining):
    def __init__(self, classification_module, destruction_module, use_cuda = True, kernel_patch = (1,1), stride_patch = (1,1)):
        super().__init__(classification_module, use_cuda = use_cuda, kernel_patch= kernel_patch, stride_patch = stride_patch)
        self.destruction_module = destruction_module

       
        self.destruction_module.kernel_update(kernel_patch, stride_patch)
        if self.use_cuda :
            self.destruction_module.cuda()
    

    
    def _train_step(self, data, target, dataset, sampling_distribution, lambda_reg = 0.0, Nexpectation = 10):
        self.classification_module.zero_grad()
        self.destruction_module.zero_grad()
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten = prepare_data_augmented(data, target, num_classes=dataset.get_category(), Nexpectation=Nexpectation, use_cuda=self.use_cuda)

        pi_list, loss_reg, _, _ = self.destruction_module(data) 
        loss_reg = lambda_reg * loss_reg


        p_z = sampling_distribution(pi_list)
        z = p_z.rsample((Nexpectation,))
        # z_reshaped = z.reshape(wanted_shape_flatten)

        y_hat = self.classification_module(data_expanded_flatten, z)
        y_hat = y_hat.reshape(Nexpectation, -1, dataset.get_category())

        log_prob_y = torch.masked_select(y_hat,one_hot_target_expanded>0.5)
        y_hat_mean = torch.mean(y_hat,0)

        neg_likelihood = - torch.mean(torch.logsumexp(log_prob_y,0)) #Size 1
        mse_loss = torch.mean(torch.sum((torch.exp(y_hat_mean)-one_hot_target.float())**2,1)) # Size 1

        loss_rec = neg_likelihood
        loss_total = loss_rec + loss_reg
        loss_total.backward()

        dic = self._create_dic(loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list)

        return dic


    def _create_dic(self, loss_total, neg_likelihood, mse_loss, loss_rec, loss_reg, pi_list):
        dic = super()._create_dic(loss_total, neg_likelihood, mse_loss)
        dic["loss_rec"] = loss_rec
        dic["loss_reg"] = loss_reg
        dic["mean_pi"] = torch.mean(torch.mean(pi_list.flatten(1),1))
        # dic["median_pi"] = torch.mean(torch.median(pi_list.flatten(1),dim=1))
        # print(pi_list.flatten(1).shape)
        # print(torch.quantile(pi_list.flatten(1),torch.tensor([0.5]),dim=1).shape)
        quantiles = torch.tensor([0.25,0.5,0.75])
        if self.use_cuda: 
            quantiles = quantiles.cuda()
        q = torch.quantile(pi_list.flatten(1),quantiles,dim=1,keepdim = True)
        dic["median_pi"] = torch.mean(q[1])
        dic["q1_pi"] = torch.mean(q[0])
        dic["q2_pi"] = torch.mean(q[2])
        return dic



    def train(self,epoch, dataset, optim_classifier, optim_destruction, sampling_distribution, lambda_reg = 0.0, Nexpectation=10):
        self.classification_module.train()
        self.destruction_module.train()
        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            
            dic = self._train_step(data, target,dataset, sampling_distribution, lambda_reg = lambda_reg, Nexpectation = Nexpectation)

            optim_classifier.step()
            optim_destruction.step()
      
            if batch_idx % 100 == 0:
                print_dic(epoch, batch_idx, dic, dataset)
            

    def test_no_var(self, dataset, sampling_distribution):
        test_loss_mse = 0
        test_loss_likelihood = 0
        correct = 0
        self.classification_module.eval()
        self.destruction_module.eval()

        with torch.no_grad():
            for data, target in dataset.test_loader:
                batch_size = data.shape[0]
                data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
                
                pi_list, _, _, _ = self.destruction_module(data, test=True)
                p_z = sampling_distribution(pi_list)


                z = p_z.sample()
                # z_reshaped = z.reshape(wanted_shape_flatten)
                y_hat = self.classification_module(data_expanded_flatten, z)

                test_loss_likelihood -= F.nll_loss(y_hat.squeeze(),target)
                test_loss_mse += torch.mean(torch.sum((torch.exp(y_hat)-one_hot_target)**2,1))
                pred = y_hat.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.unsqueeze(0).data.view_as(pred)).sum()
            test_loss_mse /= len(dataset.test_loader.dataset) * batch_size
            # test_losses.append(test_loss)
            print('\nTest set: MSE: {:.4f}, Likelihood {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                -test_loss_likelihood, test_loss_mse, correct, len(dataset.test_loader.dataset),
                100. * correct / len(dataset.test_loader.dataset)))
            print("\n")



    def MCMC(self, dataset, data, target, sampling_distribution, Niter, eps = 1e-6, burn = 1000, jump = 50):
        self.classification_module.eval()
        self.destruction_module.eval()
        with torch.no_grad():
            sample_list = []
            sample_list_readable = []

            data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
            batch_size = data.shape[0]
            input_size = data.shape[1] * data.shape[2] * data.shape[3]

            pi_list,_,_,_ = self.destruction_module(data, test = True)
            pz = sampling_distribution(pi_list)
            previous_z = pz.sample()
            # previous_z_reshaped = previous_z.reshape(wanted_shape_flatten)
            # previous_z_readable = 
            y_hat = self.classification_module(data_expanded_flatten, previous_z)
            y_hat_squeeze = y_hat.squeeze()
            previous_log_py = torch.masked_select(y_hat_squeeze, one_hot_target>0.5)

            for k in range(Niter):
                z = pz.sample()
                # z_reshaped = z.reshape(wanted_shape_flatten)

                y_hat = self.classification_module(data_expanded_flatten, z)
                log_py = torch.masked_select(y_hat_squeeze, one_hot_target>0.5)

                u = torch.rand((batch_size)) 
                if self.use_cuda:
                    u = u.cuda()
                
                proba_acceptance = torch.exp(log_py-previous_log_py)
                mask_acceptance = u<proba_acceptance
               
                mask_acceptance = mask_acceptance.unsqueeze(1).expand((-1,z.shape[1]))

                previous_z = torch.where(mask_acceptance, z, previous_z)
                # previous_z_reshaped = previous_z.reshape(wanted_shape_flatten)
                y_hat = self.classification_module(data_expanded_flatten, previous_z)
                y_hat_squeeze = y_hat.squeeze()
                previous_log_py = torch.masked_select(y_hat_squeeze, one_hot_target>0.5)

                if k > burn and k%jump == 0 :
                    sample_list.append(previous_z.cpu()[None,:,:])
                    sample_list_readable.append(self.classification_module.imputation.readable_sample(previous_z).cpu()[None, :, :])

            sample_list_readable = torch.mean(torch.cat(sample_list_readable),0)
            sample_list = torch.mean(torch.cat(sample_list),0)
            # sample_list = torch.mean(sample_list,0)

            return sample_list_readable




        


class variationalTraining(noVariationalTraining):
    def __init__(self, classification_module, destruction_module, kernel_patch = (1,1), stride_patch = (1,1)):
        super().__init__(classification_module, destruction_module, kernel_patch, stride_patch)


    

    def _prob_calc(self, y_hat, one_hot_target_expanded, z , pz, qz):
        Nexpectation = one_hot_target_expanded.shape[0]
        log_prob_y = torch.masked_select(y_hat,one_hot_target_expanded>0.5).reshape(Nexpectation,-1)
        log_prob_pz = torch.sum(pz.log_prob(z),-1)
        log_prob_qz = torch.sum(qz.log_prob(z),-1)
        
        return log_prob_y,log_prob_pz,log_prob_qz
    
    def _likelihood_var(self, y_hat, one_hot_target_expanded, z, pz, qz):
        log_prob_y, log_prob_pz, log_prob_qz = self._prob_calc(y_hat, one_hot_target_expanded, z, pz, qz)
        # print(log_prob_y.shape)
        return torch.mean(torch.logsumexp(log_prob_y+log_prob_pz-log_prob_qz,0))


    def _train_step(self, data, target, dataset, sampling_distribution, sampling_distribution_var, lambda_reg = 0.0, lambda_reg_var= 0.0, Nexpectation = 10):
        self.classification_module.zero_grad()
        self.destruction_module.zero_grad()


        batch_size = data.shape[0]
        data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), Nexpectation=Nexpectation, use_cuda=self.use_cuda)

        pi_list, loss_reg, pi_list_var, loss_reg_var = self.destruction_module(data, one_hot_target = one_hot_target, do_variational=True)
        loss_reg = lambda_reg * loss_reg
        loss_reg_var = lambda_reg_var * loss_reg_var

        pz = sampling_distribution(pi_list)
        qz = sampling_distribution_var(pi_list_var)
        
        
        z = qz.rsample((Nexpectation,))
        
        y_hat = self.classification_module(data_expanded_flatten, z).reshape((Nexpectation,batch_size, dataset.get_category()))
        y_hat_mean = torch.mean(y_hat, 0)


        neg_likelihood = - self._likelihood_var(y_hat,one_hot_target_expanded, z, pz, qz)
        mse_loss = torch.mean(torch.sum((torch.exp(y_hat_mean)-one_hot_target)**2,1))

        loss_rec = neg_likelihood
        loss_total = loss_rec + loss_reg + loss_reg_var
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
        dic["loss_reg_var"] = loss_reg_var
        dic["mean_pi_var"] = torch.mean((torch.mean(pi_list_var.squeeze(),1))).item()
        quantiles = torch.tensor([0.25,0.5,0.75])
        if self.use_cuda: 
            quantiles = quantiles.cuda()
        q = torch.quantile(pi_list_var.flatten(1),quantiles,dim=1,keepdim = True)
        dic["median_pi_var"] = torch.mean(q[1])
        dic["q1_pi_var"] = torch.mean(q[0])
        dic["q2_pi_var"] = torch.mean(q[2])
        dic["mean pi diff"] = torch.mean(
            torch.mean(
                (pi_list.flatten(1)-pi_list_var.flatten(1))**2,
                1
                )
            ).item()
        return dic

    def train(self, epoch, dataset, optim_classifier, optim_destruction, sampling_distribution, sampling_distribution_var, lambda_reg = 0.0, lambda_reg_var= 0.0, Nexpectation = 10):
        self.classification_module.train()
        self.destruction_module.train()

        for batch_idx, (data, target) in enumerate(dataset.train_loader):

            dic = self._train_step(
                data, target, dataset,
                sampling_distribution, sampling_distribution_var,
                lambda_reg = lambda_reg, lambda_reg_var= lambda_reg_var,
                Nexpectation = Nexpectation
                )
                    
            if batch_idx % 100 == 0:
                print_dic(epoch, batch_idx, dic, dataset)
            
            optim_classifier.step()
            optim_destruction.step()
        
        return dic
            # break

            
    def test_var(self, dataset, sampling_distribution, sampling_distribution_var):
        self.classification_module.eval()
        self.destruction_module.eval() 


        test_loss_mse = 0
        test_loss_likelihood = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in dataset.test_loader:
                batch_size = data.shape[0]
                data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)


                pi_list, _, pi_list_var, _ = self.destruction_module(data, one_hot_target=one_hot_target, do_variational=True, test=True)
                pz = sampling_distribution(pi_list)
                qz = sampling_distribution_var(pi_list_var)

                z = qz.sample()
                # z_reshaped = z.reshape((wanted_shape_flatten))

                y_hat = self.classification_module(data_expanded_flatten, z)
                y_hat_squeeze = y_hat.squeeze()

                test_loss_likelihood = self._likelihood_var(y_hat,one_hot_target_expanded,z,pz,qz)
                test_loss_mse += torch.sum(torch.sum((torch.exp(y_hat_squeeze)-one_hot_target)**2,1))
                pred = y_hat_squeeze.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

            test_loss_mse /= len(dataset.test_loader.dataset) * batch_size
            # test_losses.append(test_loss)
            print('\nTest set: AMSE: {:.4f}, Likelihood {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss_likelihood, test_loss_mse, correct, len(dataset.test_loader.dataset),
                100. * correct / len(dataset.test_loader.dataset)))
            print("\n")


    def MCMC_var(self, dataset, data, target, sampling_distribution, sampling_distribution_var, Niter, eps = 1e-6, burn = 1000, jump = 50):
        self.classification_module.eval()
        self.destruction_module.eval()
        with torch.no_grad():
            sample_list = []
            sample_list_readable = []

            data, target, one_hot_target, one_hot_target_expanded, data_expanded_flatten, wanted_shape_flatten  = prepare_data_augmented(data, target, num_classes=dataset.get_category(), use_cuda=self.use_cuda)
            batch_size = data.shape[0]
            input_size = data.shape[1] * data.shape[2] * data.shape[3]

            pi_list,_,pi_list_var,_ = self.destruction_module(data, one_hot_target= one_hot_target, do_variational=True, test = True)
            pz = sampling_distribution(pi_list)
            qz = sampling_distribution_var(pi_list_var)
            previous_z = qz.sample()
     
            # previous_z_reshaped = previous_z.reshape(wanted_shape_flatten)

            y_hat = self.classification_module(data_expanded_flatten, previous_z)
            previous_log_py, previous_log_pz, previous_log_qz = self._prob_calc(y_hat, one_hot_target_expanded, previous_z, pz, qz)
            # print(previous_log_py.shape)

            for k in range(Niter):
                z = qz.sample()
                # z_reshaped = z.reshape(wanted_shape_flatten)

                y_hat = self.classification_module(data_expanded_flatten, z)
                log_py, log_pz, log_qz = self._prob_calc(y_hat, one_hot_target_expanded, z, pz, qz)

                u = torch.rand((batch_size)) 
                if self.use_cuda:
                    u = u.cuda()
                

                proba_acceptance = torch.exp((log_py + log_pz - log_qz) - (previous_log_py + previous_log_pz - previous_log_qz)).squeeze()
                mask_acceptance = u<proba_acceptance
                mask_acceptance = mask_acceptance.unsqueeze(1).expand((batch_size,z.shape[1]))
                previous_z = torch.where(mask_acceptance, z, previous_z)
                # previous_z_reshaped = previous_z.reshape(wanted_shape_flatten)
                y_hat = self.classification_module(data_expanded_flatten, previous_z)
                y_hat_squeeze = y_hat.squeeze()
                previous_log_py, previous_log_pz, previous_log_qz = self._prob_calc(y_hat, one_hot_target_expanded, previous_z, pz, qz)

                if k > burn and k%jump == 0 :
                    sample_list.append(previous_z.cpu()[None,:,:])
                    sample_list_readable.append(self.classification_module.imputation.readable_sample(previous_z).cpu()[None, :, :])

            sample_list_readable = torch.mean(torch.cat(sample_list_readable),0)
            sample_list = torch.mean(torch.cat(sample_list),0)

            return sample_list_readable