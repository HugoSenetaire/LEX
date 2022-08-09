import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn

from ..utils.utils import *

def eval_selection(interpretable_module, loader, args):
    interpretable_module.prediction_module.imputation.nb_imputation_mc_test = 1
    interpretable_module.prediction_module.imputation.nb_imputation_iwae_test = 1     
    interpretable_module.eval()
    
    if args.args_selection.rate is not None :
        rate = args.args_selection.rate
        if rate > 0.:
            dic = eval_selection_local(interpretable_module, loader, rate)
        else :
            dic = eval_selection_local(interpretable_module, loader,)
    else :
        dic = eval_selection_local(interpretable_module, loader,)
    return dic



def eval_selection_local(interpretable_module, loader, rate = None):
    dic = {}
    dic["selection_auroc"] = 0
    sum_error_round = 0
    sum_error = 0
    error_selection_auroc = False

    nb_data = len(loader.test_loader.dataset)

    fp = np.zeros((nb_data,), dtype=np.float32)
    tp = np.zeros((nb_data,), dtype=np.float32)
    fn = np.zeros((nb_data,), dtype=np.float32)
    tn = np.zeros((nb_data,), dtype=np.float32)
    total_pi_list = np.zeros((nb_data,11,), dtype=np.float32)

    for batch in loader.test_loader :
        try :
            data, target, index = batch
        except :
            print("Should give index to get the eval selection")
            return {}

        if not hasattr(loader.dataset, "optimal_S_test") :
            raise AttributeError("This dataset do not have an optimal S defined")
        else :
            optimal_S_test = loader.dataset.optimal_S_test[index]

        X_test = data.type(torch.float32)
        Y_test = target.type(torch.float32)
        if next(interpretable_module.parameters()).is_cuda :
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
            optimal_S_test = optimal_S_test.cuda()

        if hasattr(interpretable_module, "selection_module"):
            selection_module = interpretable_module.selection_module
            distribution_module = interpretable_module.distribution_module
            selection_module.eval()
            distribution_module.eval()
            selection_evaluation = True
        else :
            print("No selection evaluation if there is no selection module")
            return {}
        prediction_module = interpretable_module.prediction_module
        prediction_module.eval()


        with torch.no_grad():
            log_pi_list, _ = selection_module(X_test)
            distribution_module(torch.exp(log_pi_list))
            z = distribution_module.sample((1,))
            z = interpretable_module.reshape(z)
            if isinstance(selection_module.activation, torch.nn.LogSoftmax):
                pi_list = distribution_module.sample((100,)).mean(dim = 0)
            else :
                pi_list = torch.exp(log_pi_list)
            
            pi_list = interpretable_module.reshape(pi_list).flatten(1)
        total_pi_list[index] = pi_list.detach().cpu().numpy()

        optimal_S_test = optimal_S_test.detach().cpu().numpy()
        if rate is None :
            sel_pred = (pi_list >0.5).detach().cpu().numpy().astype(int)
        else :
            dim_pi_list = np.prod(interpretable_module.selection_module.selector.output_size)
            k = max(int(dim_pi_list * rate),1)
            top_k_value, to_sel = torch.topk(pi_list.flatten(1), k, dim = 1)
            # min_value = torch.min(pi_list.flatten(1)[to_sel],dim=1)
            # print(min_value[0].shape)
            # sel_pred =  pi_list.flatten(1) > min_value
            # aux = torch.min(top_k_value,dim=1)[0].unsqueeze(1).expand(pi_list.shape[0],pi_list.shape[1])
            # print(torch.sum(torch.where(pi_list.flatten(1) >aux , torch.tensor(1), torch.tensor(0))))
            sel_pred = torch.zeros_like(pi_list).scatter(1, to_sel, 1).detach().cpu().numpy().astype(int)

        sel_true = optimal_S_test.reshape(sel_pred.shape)


        fp[index] = np.sum((sel_pred == 1) & (sel_true == 0),axis=-1)
        tp[index] = np.sum((sel_pred == 1) & (sel_true == 1),axis=-1)
        fn[index] = np.sum((sel_pred == 0) & (sel_true == 1),axis=-1)
        tn[index] = np.sum((sel_pred == 0) & (sel_true == 0),axis=-1)

       


        sum_error_round += np.sum(np.abs(optimal_S_test.reshape(-1) - sel_pred.reshape(-1)))
        sum_error += np.sum(np.abs(optimal_S_test.reshape(-1) - pi_list.reshape(-1).detach().cpu().numpy()))
        try :
            dic["selection_auroc"] += sklearn.metrics.roc_auc_score(optimal_S_test.reshape(-1), pi_list.reshape(-1))
        except :
            error_selection_auroc = True


    dic["fp_total"] = np.sum(fp)
    dic["tp_total"] = np.sum(tp)
    dic["fn_total"] = np.sum(fn)
    dic["tn_total"] = np.sum(tn)
    dic["fpr_total"] = dic["fp_total"] / (dic["fp_total"] + dic["tn_total"] + 1e-8)
    dic["tpr_total"] = dic["tp_total"] / (dic["tp_total"] + dic["fn_total"] + 1e-8)
    dic["fdr_total"] = dic["fp_total"] / (dic["fp_total"] + dic["tp_total"] + 1e-8)

    total = nb_data
    
    fpr = fp / (fp + tn + 1e-8)
    tpr = tp / (tp + fn + 1e-8)
    fdr = fp / (fp + tp + 1e-8)

    dic["fpr_mean"] = np.mean(fpr)
    dic["tpr_mean"] = np.mean(tpr)
    dic["fdr_mean"] = np.mean(fdr)
    dic["fpr_std"] = np.std(fpr)
    dic["tpr_std"] = np.std(tpr)
    dic["fdr_std"] = np.std(fdr)

    dic["selection_accuracy_rounded"] = sum_error_round / total
    dic["selection_accuracy"] = sum_error / total

    if error_selection_auroc :
        dic["selection_auroc"] = -1
    else :
        dic["selection_auroc"] /= len(loader.test_loader)

    if rate is not None :
        print("Selection Test rate {:.3f} : fdr_mean {:.4f} fpr_mean {:.4f} tpr_mean {:.4f} fdr_std {:.4f} fpr_std {:.4f} tpr_std {:.4f}".format(rate, dic["fdr_mean"], dic["fpr_mean"], dic["tpr_mean"], dic["fdr_std"], dic["fpr_std"], dic["tpr_std"]))
    else :
        print("Selection Test : fdr_mean {:.4f} fpr_mean {:.4f} tpr_mean {:.4f} fdr_std {:.4f} fpr_std {:.4f} tpr_std {:.4f}".format(dic["fdr_mean"], dic["fpr_mean"], dic["tpr_mean"], dic["fdr_std"], dic["fpr_std"], dic["tpr_std"]))

    if rate is not None :
        print("Selection Test rate {:.3f} : fdr {:.4f} fpr {:.4f} tpr {:.4f} auroc {:.4f} accuracy {:.4f}".format(rate, dic["fdr_total"], dic["fpr_total"], dic["tpr_total"], dic["selection_auroc"], dic["selection_accuracy"]))
    else :
        print("Selection Test : fdr {:.4f} fpr {:.4f} tpr {:.4f} auroc {:.4f} accuracy {:.4f}".format(dic["fdr_total"], dic["fpr_total"], dic["tpr_total"], dic["selection_auroc"], dic["selection_accuracy"]))
    
    mean_total_pi_list = np.mean(total_pi_list, axis = 0)
    std_total_pi_list = np.std(total_pi_list, axis = 0)

    for dim in range(11):
        dic[f"Mean_pi_{dim}"] = mean_total_pi_list[dim]
        dic[f"Std_pi_{dim}"] = std_total_pi_list[dim]

    print("Pi_list Mean {}, Std {}".format(mean_total_pi_list, std_total_pi_list))

    return dic