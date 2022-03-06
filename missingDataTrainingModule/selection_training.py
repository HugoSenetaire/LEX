from missingDataTrainingModule.Classification import classification_module
from .utils import *
import torch.nn.functional as F


class selectionTraining():
    def __init__(self, selection_module, use_reg = False):    
        self.selection_module = selection_module
        self.use_reg = use_reg

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
            self.selection_module = self.selection_module.cuda()
            self.use_cuda = True
       

    def _create_dic(self, mse_loss, loss_reg = None):
        dic = {}
        dic["mse_loss"] = mse_loss.item()
        if loss_reg is not None :
            dic["loss_reg"] = loss_reg.item()
        else :
            dic["loss_reg"] = 0
        return dic

    def _create_dic_test(self, correct, mse_loss):
        dic = {}
        dic["accuracy"] = correct.item()
        dic["mse"] = mse_loss.item()
        return dic

    def zero_grad(self):
        self.selection_module.zero_grad()

    def train(self):
        self.selection_module.train()


    def _train_step(self, data, target, dataset, index = None):
        self.zero_grad()
        if self.use_cuda :
            data, target, index = on_cuda(data, target, index)
        
        log_pi_list, loss_reg = self.selection_module(data,)

        mse_loss = F.mse_loss(torch.exp(log_pi_list), target, reduction='mean')

        if self.use_reg :
            dic = self._create_dic(mse_loss, loss_reg)
            (mse_loss + loss_reg).backward()
        else :
            dic = self._create_dic(mse_loss, )
            mse_loss.backward()

        self.optim_selection.step()
        return dic


    def train_epoch(self, epoch, loader,  save_dic = False, verbose = False,):
        self.train()

        total_dic = {}
        for batch_idx, data in enumerate(loader.train_loader):
            input, _, index = parse_batch(data)
            target = loader.dataset.optimal_S_train[index].type(torch.float32)
            dic = self._train_step(input, target, loader.dataset, index=index)

            if batch_idx % 100 == 0 :
                if verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
        
        if self.scheduler_selection is not None :
            print(f"Learning Rate selection : {self.scheduler_selection.get_last_lr()}")
            self.scheduler_selection.step()
        
        return total_dic



    def test(self, epoch, loader):
        self.selection_module.eval()

        dataset = loader.dataset
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_index, data in enumerate(loader.test_loader):
                data, _, index = parse_batch(data)
                target = loader.dataset.optimal_S_test[index].type(torch.float32)
                if self.use_cuda :
                    data, target, index = on_cuda(data, target, index)
                log_pi_list, _ = self.selection_module(data,)
                mse = F.mse_loss(torch.exp(log_pi_list), target)
                test_loss += mse
                pred = torch.exp(log_pi_list).data.round()
                correct_current = pred.eq(target.data.view_as(pred)).sum()
                correct += correct_current


        test_loss /= len(loader.test_loader.dataset)
        print('\n Epoch {}'.format(epoch))
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.test_loader.dataset) * np.prod(loader.dataset.get_dim_input()),
        100. * correct / len(loader.test_loader.dataset) / np.prod(loader.dataset.get_dim_input())))
        total_dic = self._create_dic_test(correct/len(loader.test_loader.dataset), test_loss)
        return total_dic

