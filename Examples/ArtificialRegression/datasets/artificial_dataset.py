import torch
import numpy as np 




torch.pi = torch.tensor(3.1415)



class ArtificialDataset():
    def __init__(self, nb_sample_train = 20, nb_sample_test = 20, give_index = False, noise_function = None, train_seed = 0, test_seed = 1, **kwargs) -> None:
        self.nb_sample_train = nb_sample_train
        self.nb_sample_test = nb_sample_test
        self.give_index = give_index
        self.noise_function = noise_function
        self.batch_size_calculate = 100
        self.optimal_S_train = None
        self.optimal_S_test = None
        self.train_seed = train_seed
        self.test_seed = test_seed

    def get_true_output(self, value, mask = None, index=None, dataset_type = None):
        raise NotImplementedError("Using abstract class ArtificialDataset")

    def get_dim_input(self):
        return (self.dim_input, )

    def get_dim_output(self):
        return self.nb_classes

    def impute(self, value, mask, index=None, dataset_type = None):
        raise NotImplementedError("Using abstract class ArtificialDataset")

    def calculate_true_selection_variation(self, X, normalize = False, classifier = None, nb_imputation = 100,):
        """
        Calculate the true selection for the given data X using the true definition of the dataset. If classifier is None, uses the true definition of output from the dataset.
        """
        output_S = np.zeros(X.shape)

        if classifier is None :
            classifier = lambda x: self.get_true_output(x)
        
        nb_batch = X.shape[0] // self.batch_size_calculate + 1

        for k in range(nb_batch) :
            if k != nb_batch - 1 :
                X_batch = X[k*self.batch_size_calculate:(k+1)*self.batch_size_calculate]
            else :
                X_batch = X[k*self.batch_size_calculate:]

        
            if k*self.batch_size_calculate == len(X):
                continue

            batch_size = X_batch.shape[0]

            X_batch = X_batch.unsqueeze(0).expand(torch.Size((nb_imputation,))+X_batch.shape).flatten(0,1)
            if output_S.shape[-1] == 2 :
                mask = torch.ones_like(X_batch, dtype=torch.float32)

                mask_firstdim = mask.clone()
                mask_firstdim[:,0] = torch.zeros(X_batch.shape[0], dtype=torch.float32)
                

                X_batch_first_dim = self.impute(value = X_batch, mask = mask_firstdim, index = None, dataset_type = None)
                Y_first_dim = classifier(X_batch_first_dim).reshape(nb_imputation, batch_size, self.nb_classes)


                mask_seconddim = mask.clone()
                mask_seconddim[:,1] = torch.zeros(X_batch.shape[0], dtype=torch.float32)
                X_batch_second_dim = self.impute(value = X_batch, mask= mask_seconddim, index = None, dataset_type = None)
                Y_second_dim = classifier(X_batch_second_dim).reshape(nb_imputation, batch_size, self.nb_classes)

                                
                true_selection_firstdim = torch.mean(torch.std(Y_first_dim, dim=0), axis=-1)
                true_selection_seconddim = torch.mean(torch.std(Y_second_dim, dim=0), axis=-1)
        
                if k != nb_batch - 1 :
                    output_S[k*self.batch_size_calculate:(k+1)*self.batch_size_calculate,0] = true_selection_firstdim.detach().cpu().numpy()
                    output_S[k*self.batch_size_calculate:(k+1)*self.batch_size_calculate,1] = true_selection_seconddim.detach().cpu().numpy()                
                else :
                    output_S[k*self.batch_size_calculate:,0] = true_selection_firstdim.detach().cpu().numpy()
                    output_S[k*self.batch_size_calculate:,1] = true_selection_seconddim.detach().cpu().numpy()   
                

            else :
                raise NotImplementedError("Not implemented for more than 2 dimensions") 

        output_S = torch.tensor(output_S).type(torch.float32)
        if normalize :
            output_S /= torch.max(output_S)

        return output_S









