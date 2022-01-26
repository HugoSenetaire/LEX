from .utils_missing import *
import torch.nn.functional as F




class selectionTraining():
    def __init__(self, selection_module,):    
        self.selection_module = selection_module
        self.compiled = False
        self.use_cuda = False


    def compile(self, optim_selection, scheduler_selection = None,):
        self.optim_selection = optim_selection
        self.scheduler_selection = scheduler_selection
        self.compiled = True

    def cuda(self):
        if not torch.cuda.is_available() :
            print("CUDA not found, using cpu instead")
        else :
            self.selection_module.cuda()
            self.use_cuda = True
       

    def _create_dic(self, mse_loss):
        dic = {}
        dic["mse_loss"] = mse_loss.item()
        return dic

    def _create_dic_test(self, mse_loss):
        dic = {}
        dic["mse"] = mse_loss.item()
        return dic

    def zero_grad(self):
        self.selection_module.zero_grad()

    def train(self):
        self.selection_module.train()


    def _train_step(self, data, target, dataset, index = None):
        self.zero_grad()
        data, _, _ = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
        log_pi_list, _ = self.selection_module(data, index= index)

        mse_loss = F.mse_loss(log_pi_list, target)
        # loss = neg_likelihood
        dic = self._create_dic(mse_loss, )
        mse_loss.backward()
        self.optim_selection.step()
        
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
        
        if self.scheduler_selection is not None :
            print(f"Learning Rate selection : {self.scheduler_selection.get_last_lr()}")
            self.scheduler_selection.step()
        
        return total_dic



    def test(self,loader):
        self.selection_module.eval()

        dataset = loader.dataset
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, target, index = parse_batch(data)

                data, target, one_hot_target = prepare_data(data, target, num_classes=dataset.get_dim_output(), use_cuda=self.use_cuda)
                log_pi_list, _ = self.selection_module(data, index = index)
                mse = F.mse_loss(log_pi_list, one_hot_target)
                
                test_loss += mse
                pred = log_pi_list.data.max(1, keepdim=True)[1]
                if self.use_cuda:
                    correct_current = pred.eq(target.cuda().data.view_as(pred)).sum()
                else :
                    correct_current = pred.eq(target.data.view_as(pred)).sum()
                correct += correct_current


        test_loss /= len(loader.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.test_loader.dataset),
        100. * correct / len(loader.test_loader.dataset)))
        total_dic = self._create_dic_test(correct/len(loader.test_loader.dataset), test_loss, test_loss)
        return total_dic

