from ..utils import parse_batch, print_dic, save_dic_helper

import numpy as np

class classic_train_epoch():
    def __init__(self, save_dic=False, verbose=True, **kwargs):
        self.save_dic = save_dic
        self.verbose = verbose
        
    def __call__(self, epoch,  loader,  trainer,):

        if hasattr(trainer, 'fix_classifier_parameters'):
            if trainer.fix_classifier_parameters and trainer.fix_selector_parameters :
                raise AttributeError("You can't train if both classifiers and selectors are fixed.")
            assert(trainer.compiled)
        trainer.interpretable_module.train()
        total_dic = {}
        print_batch_every = max(len(loader.dataset_train)//loader.train_loader.batch_size//10, 1)

        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            dic = trainer._train_step(data, target, loader.dataset, index=index, need_dic = (batch_idx % print_batch_every == 0))
            if batch_idx % print_batch_every == 0 :
                if self.verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if self.save_dic :
                    total_dic = save_dic_helper(total_dic, dic)

        trainer.scheduler_step()    
        return total_dic


class alternate_fixing_train_epoch():
    def __init__(self, save_dic= False, verbose= True, nb_step_fixed_classifier = 1, nb_step_fixed_selector = 1, nb_step_all_free = 1,) :
        super().__init__(save_dic, verbose)
        assert np.any(np.array([nb_step_fixed_classifier, nb_step_fixed_selector, nb_step_all_free])>0)
        self.nb_step_fixed_classifier = nb_step_fixed_classifier
        self.nb_step_fixed_selector = nb_step_fixed_selector
        self.nb_step_all_free = nb_step_all_free


    def __call__(self, epoch, loader, trainer):
        if trainer.fix_classifier_parameters and self.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(trainer.compiled)
        trainer.interpretable_module.train()
        total_dic = {}
        print_batch_every =  max(len(loader.dataset_train)//loader.train_loader.batch_size//10, 1)

        total_program = self.nb_step_fixed_classifier + self.nb_step_fixed_selector + self.nb_step_all_free
        init_number = np.random.randint(0, total_program+1)
        for batch_idx, data in enumerate(loader.train_loader):
            data, target, index = parse_batch(data)
            if (batch_idx + init_number) % total_program < self.nb_step_fixed_classifier :
                trainer.fix_classifier_parameters = True
                trainer.fix_selector_parameters = False
            elif (batch_idx + init_number) % total_program < self.nb_step_fixed_classifier + self.nb_step_fixed_selector :
                trainer.fix_classifier_parameters = False
                trainer.fix_selector_parameters = True
            else :
                trainer.fix_classifier_parameters = False
                trainer.fix_selector_parameters = False

            dic = trainer._train_step(data, target, loader.dataset, index=index, need_dic= (batch_idx % print_batch_every == 0))
            if batch_idx % print_batch_every == 0 :
                if self.verbose :
                    print_dic(epoch, batch_idx, dic, loader)
                if self.save_dic :
                    total_dic = save_dic_helper(total_dic, dic)
                    trainer.scheduler_step()    
        return total_dic

    
class alternate_ordinary_train_epoch():
    def __init__(self, save_dic= False, verbose= True, ratio_class_selection = 2, ordinaryTraining=None,) :
        super().__init__(save_dic, verbose)


        if ordinaryTraining is None :
            raise AttributeError("ratio_class_selection is not None but ordinaryTraining is None, nothing is defined for the ratio")
        else :
            if not ordinaryTraining.compiled :
                raise AttributeError("ratio_class_selection is not None but ordinaryTraining is not compiled")
        assert ratio_class_selection>0

        self.ordinaryTraining = ordinaryTraining
        self.ratio_class_selection = ratio_class_selection

    def __call__(self, epoch, loader, trainer):
        if trainer.fix_classifier_parameters and trainer.fix_selector_parameters :
            raise AttributeError("You can't train if both classifiers and selectors are fixed.")
        assert(trainer.compiled)
        trainer.interpretable_module.train()
        total_dic = {}
        print_batch_every = max(len(loader.dataset_train)//loader.train_loader.batch_size//10, 1)

        if self.ratio_class_selection >=1 :
            ratio_step = max(np.round(self.ratio_class_selection), 1)
            init_number = np.random.randint(0, ratio_step+1) # Just changing which part of the dataset is used for ordinary training and the others. TODO, there might be a more interesting way to do this, inside the dataloader for instance ?
            for batch_idx, data in enumerate(loader.train_loader):
                    data, target, index = parse_batch(data)
                    if (batch_idx + init_number) % (ratio_step+1) == 0 :
                        dic = self._train_step(data, target, loader.dataset, index=index, need_dic= (batch_idx % print_batch_every == 0))
                        if batch_idx % print_batch_every == 0 :
                            if self.verbose :
                                print_dic(epoch, batch_idx, dic, loader)
                            if self.save_dic :
                                total_dic = save_dic_helper(total_dic, dic)
                    else :
                        dic = self.ordinaryTraining._train_step(data, target, loader.dataset, index=index,)
        else :
            step_only_pred = np.round(1./self.ratio_class_selection).astype(int)
            init_number = np.random.randint(0, step_only_pred+1)
            for batch_idx, data in enumerate(loader.train_loader):
                data, target, index = parse_batch(data)
                if (batch_idx + init_number) % (step_only_pred+1) == 0 :
                    dic = self.ordinaryTraining._train_step(data, target, loader.dataset, index=index, )
                    if batch_idx % print_batch_every == 0 :
                        if self.verbose :
                            print_dic(epoch, batch_idx, dic, loader)
                else :
                    dic = self._train_step(data, target, loader.dataset, index=index,  need_dic= (batch_idx % print_batch_every == 0))
                    if batch_idx % print_batch_every == 0 :
                        if self.verbose :
                            print_dic(epoch, batch_idx, dic, loader)
                        if self.save_dic :
                            total_dic = save_dic_helper(total_dic, dic)
        
        trainer.scheduler_step()    
        return total_dic
