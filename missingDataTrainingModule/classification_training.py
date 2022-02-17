from missingDataTrainingModule import PytorchDistributionUtils
from .utils_missing import *
import torch.nn.functional as F




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

    def _calculate_neg_likelihood(self, data, index, log_y_hat, target,):
        """
        Calculate the negative log likelihood of the classification per element of the batch, no reduction is done. 
        The calculation depends on the mode (post hoc, argmax post hoc, or none)

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
            if self.post_hoc_guidance is not None :
                out_y, _ = self.post_hoc_guidance(data, index = index)
            else :
                raise AttributeError("You can't have post-hoc without a post hoc guidance if you are only training the classification")
            out_y = out_y.detach()
            if self.argmax_post_hoc :
                out_y = torch.argmax(out_y, -1)
                neg_likelihood = F.nll_loss(log_y_hat, out_y, reduce = False)
            else :
                neg_likelihood = - torch.sum(torch.exp(out_y) * log_y_hat, -1)

        return neg_likelihood


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
       

    def _create_dic(self,loss, neg_likelihood,):
        dic = {}
        dic["likelihood"] = -neg_likelihood.item()
        dic["total_loss"] = loss.item()
        return dic

    def _create_dic_test(self, correct, neg_likelihood, mse_loss):
        dic = {}
        dic["accuracy_prediction_no_selection"] = correct.item()
        dic["likelihood"] = -neg_likelihood.item()
        dic["mse"] = mse_loss.item()
        return dic

    def zero_grad(self):
        self.classification_module.zero_grad()

    def train(self):
        self.classification_module.train()


    def _train_step(self, data, target, dataset, index = None):
        self.zero_grad()
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        log_y_hat, _ = self.classification_module(data, index= index)

        # neg_likelihood = F.nll_loss(log_y_hat, target)
        neg_likelihood = self._calculate_neg_likelihood(data, index, log_y_hat, target)
        loss = torch.mean(neg_likelihood)
        dic = self._create_dic(loss, torch.mean(neg_likelihood), )
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
                data, target, index = parse_batch(data)
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



class trueSelectionTraining(ordinaryTraining):
    def __init__(self, classification_module, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False,):   
        super().__init__(classification_module, post_hoc, post_hoc_guidance, argmax_post_hoc)

    def _train_step(self, data, true_mask, target, dataset, index=None):
        self.zero_grad()

        
        data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        true_mask.to(data.device)
        log_y_hat, _ = self.classification_module(data, index= index, mask = true_mask)

        # neg_likelihood = F.nll_loss(log_y_hat, target)
        neg_likelihood = self._calculate_neg_likelihood(data, index, log_y_hat, target)
        loss = torch.mean(neg_likelihood)
        dic = self._create_dic(loss, torch.mean(neg_likelihood), )
        loss.backward()
        self.optim_classification.step()
        return dic

    def train_epoch(self, epoch, loader,  save_dic = False, verbose = False,):
        self.train()

        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            true_mask = loader.dataset.optimal_S_train[index].type(torch.float32).to(data.device)
            dic = self._train_step(data, true_mask, target, loader.dataset, index=index)

            if batch_idx % 100 == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_classification is not None :
            self.scheduler_classification.step()
        
        return total_dic


class EVAL_X(ordinaryTraining):
    def __init__(self, classification_module, fixed_distribution = PytorchDistributionUtils.wrappers.FixedBernoulli(),
                reshape_mask_function = None, post_hoc = False, post_hoc_guidance = None, argmax_post_hoc = False,):
        super().__init__(classification_module, post_hoc_guidance = post_hoc_guidance, post_hoc = post_hoc, argmax_post_hoc = argmax_post_hoc)
        # print(self.classification_module)
        self.fixed_distribution = fixed_distribution
        self.reshape_mask_function = reshape_mask_function


    def train_epoch(self, epoch, loader, nb_sample_z_monte_carlo = 10, nb_sample_z_IWAE = 1, save_dic=False, verbose=False,):
        self.train()
        total_dic = {}
        print_batch_every = len(loader.dataset_train)//loader.train_loader.batch_size//10
        for batch_idx, data in enumerate(loader.train_loader):
            input, target, index = parse_batch(data)

            dic = self._train_step(input, target, loader.dataset, index=index, nb_sample_z_monte_carlo = nb_sample_z_monte_carlo, nb_sample_z_IWAE=nb_sample_z_IWAE, need_dic= (batch_idx % print_batch_every == 0))
            
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

    def calculate_cost(self, 
                    mask_expanded, #Shape is (nb_sample_z_monte_carlo, batch_size, nb_sample_z_IWAE,channel, dim...)
                    data_expanded_multiple_imputation, # Shape is (nb_imputation, nb_sample_z_monte_carlo, batch_size, nb_sample_z_IWAE,channel, dim...)
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
        else :
            index_expanded = None
        
        target_expanded_multiple_imputation_flatten = target_expanded_multiple_imputation.flatten(0,3)
        data_expanded_multiple_imputation_flatten = data_expanded_multiple_imputation.flatten(0,3)

        log_y_hat, _ = self.classification_module(data_expanded_multiple_imputation[0], mask_expanded, index = index_expanded)
        log_y_hat = log_y_hat.reshape(data_expanded_multiple_imputation.shape[:4] + torch.Size((dim_output,)))
        
        neg_likelihood = self._calculate_neg_likelihood(data = data_expanded_multiple_imputation_flatten,
                            index = None,
                            log_y_hat = log_y_hat.flatten(0,3),
                            target = target_expanded_multiple_imputation_flatten,
                            )

        neg_likelihood = neg_likelihood.reshape(data_expanded_multiple_imputation.shape[:4])
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=0)  - torch.log(torch.tensor(nb_imputation).type(torch.float32))
        neg_likelihood = torch.logsumexp(neg_likelihood, axis=-1) - torch.log(torch.tensor(nb_sample_z_IWAE).type(torch.float32))

        return neg_likelihood



    def _train_step(self, data, target, dataset, index = None, nb_sample_z_monte_carlo = 10, nb_sample_z_IWAE = 1, need_dic = False,):
        nb_imputation = self.classification_module.imputation.nb_imputation        
        batch_size = data.shape[0]
        if self.use_cuda :
            data, target, index = on_cuda(data, target = target, index = index,)
        one_hot_target = get_one_hot(target, num_classes = dataset.get_dim_output())
        data_expanded_multiple_imputation, target_expanded_multiple_imputation, index_expanded_multiple_imputation, one_hot_target_expanded_multiple_imputation = prepare_data_augmented(data, target = target, index=index, one_hot_target = one_hot_target, nb_sample_z_monte_carlo= nb_sample_z_monte_carlo,  nb_sample_z_IWAE= nb_sample_z_IWAE, nb_imputation = nb_imputation)

        # Destructive module :
        p_z = self.fixed_distribution(data_expanded_multiple_imputation[0,0],)

        # Train classification module :
        z = self.fixed_distribution.sample(sample_shape = (nb_sample_z_monte_carlo,))
        
        # Classification module :
        loss_classification = self.calculate_cost(mask_expanded = z,
                        data_expanded_multiple_imputation = data_expanded_multiple_imputation,
                        target_expanded_multiple_imputation = target_expanded_multiple_imputation,
                        index_expanded_multiple_imputation = index_expanded_multiple_imputation,
                        one_hot_target_expanded_multiple_imputation = one_hot_target_expanded_multiple_imputation,
                        dim_output = dataset.get_dim_output(),
                        )

        loss_classification = loss_classification.mean(axis = 0) # Monte Carlo average
        torch.mean(loss_classification, axis=0).backward() # Batch average
        self.optim_classification.step()

        if need_dic :
            dic = self._create_dic(loss = torch.mean(loss_classification),
                                neg_likelihood = torch.mean(loss_classification),
                        )
        else :
            dic = {}
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


