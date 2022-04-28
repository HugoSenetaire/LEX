import os
from statistics import mean
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture
import tqdm

from .vaeac import load_model, train_vaeac, GeneratorDataset, ZipDatasets, impute, save_imputed_images




class DatasetInput(Dataset):
    """CelebA dataset."""

    def __init__(self, data):
        """
        Args:
            data (torch.tensor):       torch.tensor of shape (N, C, H, W)
          
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        return image


def train_VAEAC(loader, model_dir, epochs = 20):
    """ Training a Gaussian Mixture Model on the data using sklearn. """
    print("TRAINING FROM MODEL DIR {}".format(model_dir))
    verbose = True
    num_workers = 0
    use_cuda = torch.cuda.is_available()
    model_module, model, optimizer, batch_size, vlb_scale_factor, mask_generator, validation_iwae, train_vlb = load_model(model_dir, use_cuda=use_cuda)
    sampler = model_module.sampler

    # load train and validation datasets
    train_dataset = DatasetInput(loader.dataset.data_train)
    validation_dataset = DatasetInput(loader.dataset.data_test)

    # build dataloaders on top of datasets
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_vaeac(model, model_dir, epochs, dataloader, val_dataloader, mask_generator, optimizer, validation_iwae, train_vlb, vlb_scale_factor, batch_size,  use_cuda = use_cuda, verbose = verbose)
    
    
    # Impute after training to see if training went right.
    masks = GeneratorDataset(mask_generator, validation_dataset)
    combine_dataloader = DataLoader(ZipDatasets(validation_dataset, masks), batch_size=batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=num_workers)

    out_dir = os.path.join(model_dir, 'inpainted_examples')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_samples = 5
    iterator = combine_dataloader
    if verbose:
        iterator = tqdm(iterator)

    image_num = 0
    for batch_tuple in iterator:
        batch, masks = batch_tuple
        batch_size = batch.shape[0]
        other_shape = batch.shape[1:]
        multiple_img_samples, samples_params = impute(model, sampler, batch, masks, nb_samples = num_samples)
        save_imputed_images(batch, masks, multiple_img_samples, samples_params, batch_size, num_samples, other_shape, out_dir, image_num)



class VAEACImputation(nn.Module):
  def __init__(self, model_dir, **kwargs):
    super().__init__()
    if not os.path.exists(model_dir):
      raise ValueError("There is no description of the model at {}".format(model_dir))
    
    self.use_cuda = torch.cuda.is_available()
    model_module, model, optimizer, batch_size, vlb_scale_factor, mask_generator, validation_iwae, train_vlb = load_model(model_dir, use_cuda=self.use_cuda)
    self.model = model
    self.sampler = model_module.sampler
    

  def __call__(self, data, mask, index = None,):
    """ Using the data and the mask, do the imputation and classification 
        
        Parameters:
        -----------
        data : torch.Tensor of shape (nb_imputation * batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        sampled : torch.Tensor of shape (nb_imputation, batch_size, nb_category)
            Sampled tensor from the Gaussian Mixture

    """
    with torch.no_grad() :
        reverse_mask = 1 - mask # Mask of implementation of vaeac is reverse compared to our standard
        multiple_data_samples, samples_params = impute(self.model, self.sampler, data, reverse_mask, self.use_cuda, nb_samples = 1)
        imputed_data = multiple_data_samples

    return imputed_data