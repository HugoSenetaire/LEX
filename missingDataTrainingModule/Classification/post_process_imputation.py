import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_missing import *
import numpy as np
import torch
import copy
from importlib import import_module
import matplotlib.pyplot as plt
import pandas as pd
from .vaeac import *
import inspect
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class BaseMethod():
  def __init__(self):
    self.to_train = False
    self.multiple_imputation = False

  def cuda(self):
    return 0

  def eval(self):
    return 0
  
  def train(self):
    return 0

  

class NetworkBasedPostProcess(BaseMethod):
  def __init__(self, network, to_train = False, deepcopy = False, use_cuda = True):
    BaseMethod.__init__(self)
    self.network = network
    self.to_train = to_train
    self.multiple_imputation = False
    self.use_cuda = use_cuda
    
    if deepcopy :
      self.network = copy.deepcopy(self.network)
    
    self.network = self.network.cuda()
    if not to_train :
      for param in self.network.parameters():
          param.requires_grad = False

  def cuda(self):
    self.network = self.network.cuda()

  def eval(self):
    self.network.eval()

  def train(self):
    self.network.train()

  def parameters(self):
    return self.network.parameters()

  def __call__(self, data_expanded, data_imputed, sample_b):
    raise NotImplementedError



class MultipleImputation(BaseMethod):
  def __init__(self, nb_imputation):
    super().__init__()
    self.multiple_imputation = True
    self.nb_imputation = nb_imputation
    self.eval_mode = False

  def eval(self):
    self.eval_mode = True

  def train(self):
    self.eval_mode = False

  def check_mode(self):
    if self.eval_mode :
      return 1
    else :
      return self.nb_imputation



class NetworkBasedMultipleImputation(NetworkBasedPostProcess, MultipleImputation):
  def __init__(self, network, to_train = False, deepcopy = False, nb_imputation = 3, use_cuda = True):
    NetworkBasedPostProcess.__init__(self, network, to_train=to_train, deepcopy=deepcopy)
    MultipleImputation.__init__(self, nb_imputation = nb_imputation)
    self.use_cuda = use_cuda

  def eval(self):
    NetworkBasedPostProcess.eval(self)
    MultipleImputation.eval(self)

  def train(self):
    NetworkBasedPostProcess.train(self)
    MultipleImputation.train(self)






### SAMPLE_B REGULARIZATION :
class SampleB_regularization(BaseMethod):
  def __init__(self):
    super().__init__()
    self.to_train = False

  def __call__(self, data_expanded, sample_b):
    raise NotImplementedError


class SimpleB_Regularization(SampleB_regularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):
    if self.rate > np.random.random():
      sample_b = torch.ones(data_expanded.shape).cuda()
    
    return sample_b


  
class Less_Destruction_Regularization(SampleB_regularization):
  def __init__(self, rate = 0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):
    sample_b = torch.where(
      ((sample_b<0.5) * torch.rand(sample_b.shape, device = "cuda")>self.rate),
      torch.zeros(sample_b.shape,device = "cuda"),
      sample_b
    )
    return sample_b

class Complete_Inversion_Regularization(SampleB_regularization):
  def __init__(self, rate=0.5):
    super().__init__()
    self.rate = rate

  def __call__(self, data_expanded, sample_b):

    sample_b = torch.where(
      (torch.rand(sample_b.shape, device = "cuda")>self.rate),
      1-sample_b,
      sample_b
    )

    return sample_b



### LOSS REGULARIZATION : 

class AutoEncoderReconstructionRegularization(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)
  
  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstruced = self.network(data_imputed)
    loss =  torch.nn.functional.mse_loss(data_reconstruced, data_expanded)
    return loss
  
### POST PROCESS REGULARIZATION :

class NetworkTransform(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstructed = self.network(data_imputed)
    return data_reconstructed, data_expanded, sample_b
  

class NetworkAdd(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)


  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstructed = self.network(data_imputed)
    data_imputed = torch.cat([data_imputed,data_reconstructed],axis = 1)
    return data_reconstructed, data_expanded, sample_b
  


class NetworkTransformMask(NetworkBasedPostProcess):
  def __init__(self, network, to_train = False, use_cuda=False, deepcopy = False):
    super().__init__(network = network, to_train = to_train, use_cuda= use_cuda, deepcopy = deepcopy)

  def __call__(self, data_expanded, data_imputed, sample_b):
    data_reconstructed = data_imputed * (1-sample_b) + self.network(data_imputed) * sample_b 
    return data_reconstructed, data_expanded, sample_b

def expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation):
    wanted_transform = tuple(np.insert(-np.ones(len(data_expanded.shape),dtype = int),1, nb_imputation))
    data_imputed_expanded = data_imputed.unsqueeze(1).expand(wanted_transform).flatten(0,1)
    data_expanded_imputation = data_expanded.unsqueeze(1).expand(wanted_transform).flatten(0,1) # N_expectation, batch_size, channels, size:...
    mask_expanded = sample_b.unsqueeze(1).expand(wanted_transform).flatten(0,1)
    
    return data_imputed_expanded, data_expanded_imputation, mask_expanded



    
class DatasetBasedImputation(MultipleImputation):
  def __init__(self, dataset, nb_imputation):
    super().__init__(nb_imputation)
    self.dataset = dataset
    self.exist = hasattr(dataset, "impute_result") 
    if self.exist :
      self.nb_imputation = nb_imputation
    else :
      self.nb_imputation = 1
      print(f"There is no theoretical method for multiple imputation with {dataset}. DatasetBasedImputation is bypassed from now on.")
    
      

  def __call__(self, data_expanded, data_imputed, sample_b):
    if self.exist :
      if not self.eval_mode :
        imputation_number = self.nb_imputation
      else :
        imputation_number = 1


      data_imputed, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, imputation_number)
      data_expanded = data_expanded.flatten(0,1)
      if len(data_expanded.shape)>2:
        data_imputed = data_imputed.flatten(0,1)
      if len(sample_b.shape)>2:
        sample_b = sample_b.flatten(0,1)

      imputed_output = self.dataset.impute_result(mask = sample_b.clone().detach(),value =  data_imputed.clone().detach())
      return imputed_output, data_expanded, sample_b
    else :
      return data_imputed, data_expanded, sample_b

  
def load_VAEAC(path_model):
  # import the module with the model networks definitions,
  # optimization settings, and a mask generator
  model_module = import_module(path_model + '.model')
  # build VAEAC on top of the imported networks
  model = VAEAC(
      model_module.reconstruction_log_prob,
      model_module.proposal_network,
      model_module.prior_network,
      model_module.generative_network
  )
  mask_generator = model_module.mask_generator
  sampler = model_module.sampler

  if  not os.path.exists(os.path.join(path_model, 'last_checkpoint.tar')):
    print("model has not been trained")
    raise NotImplementedError
  location = 'cuda'
  checkpoint = torch.load(os.path.join(path_model, 'last_checkpoint.tar'),
                          map_location=location)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  # validation_iwae = checkpoint['validation_iwae']
  # train_vlb = checkpoint['train_vlb']
  return model, sampler


# class VAEAC_Imputation(NetworkBasedPostProcess):
#   def __init__(self, network, sampler, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
#     super().__init__(network= network, to_train=to_train, use_cuda=use_cuda, deepcopy=deepcopy)
#     self.nb_imputation = nb_imputation
#     self.sampler = sampler
#     self.multiple_imputation = True
#     raise NotImplementedError
#   def __call__(self, data_expanded, data_imputed, sample_b):
#     batch = data_imputed
#     masks = 1-sample_b
#     init_shape = batch.shape[0]
#     if torch.cuda.is_available():
#         batch = batch.cuda()
#         masks = masks.cuda()

        
#     if not self.eval_mode :
#       nb_imputation = self.nb_imputation
#     else :
#       nb_imputation = 1
      


#     # compute imputation distributions parameters
#     samples_params = self.network.generate_samples_params(batch,
#                                                   masks,
#                                                   nb_imputation)

    
#     img_samples = self.sampler(samples_params, multiple = True)



    
#     _, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)

#     new_data = img_samples *  (1-mask_expanded) + data_expanded * mask_expanded 
#     new_data = new_data.flatten(0,1)
#     return new_data, data_expanded, mask_expanded

class VAEAC_Imputation_DetachVersion(NetworkBasedMultipleImputation):
  def __init__(self, network, sampler, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(network= network, to_train=to_train, use_cuda=use_cuda, deepcopy=deepcopy)
    self.nb_imputation = nb_imputation
    self.sampler = sampler
    self.multiple_imputation = True
    


  def __call__(self, data_expanded, data_imputed, sample_b, show_output = False):
    batch = data_imputed
    masks = 1-sample_b
    init_shape = batch.shape[0]


    if torch.cuda.is_available():
        batch = batch.cuda()
        masks = masks.cuda()


    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1
    

  
    with torch.no_grad():

      # compute imputation distributions parameters
      samples_params = self.network.generate_samples_params(batch.detach(),
                                                    masks.detach(),
                                                    nb_imputation,
                                                    need_observed = False)
      # img_samples = []
      # for element in samples_params :
      img_samples = self.sampler(samples_params, multiple = True).flatten(0,1)



    
    _, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    new_data = img_samples.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 
    # if np.random.rand()<0.01:
    #   fig, axs = plt.subplots(3,2)
    #   axs[0,0].imshow(batch[0].reshape(28,28).detach().cpu().numpy(), cmap = 'gray')
    #   axs[0,1].imshow(masks[0].reshape(28,28).detach().cpu().numpy(), cmap = 'gray')
    #   axs[1,0].imshow(img_samples[0].reshape(28,28).detach().cpu().numpy(), cmap='gray')
    #   axs[1,1].imshow(new_data[0].reshape(28,28).detach().cpu().numpy(), cmap='gray')
    #   axs[2,0].imshow(img_samples[1].reshape(28,28).detach().cpu().numpy(), cmap='gray')
    #   axs[2,1].imshow(new_data[1].reshape(28,28).detach().cpu().numpy(), cmap='gray')
    #   plt.show()
      # plt.close(fig)
    return new_data, data_expanded, sample_b

class VAEAC_Imputation_Renormalized(NetworkBasedMultipleImputation):
  def __init__(self, network, sampler, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(network= network, to_train=to_train, use_cuda=use_cuda, deepcopy=deepcopy)
    self.nb_imputation = nb_imputation
    self.sampler = sampler
    self.multiple_imputation = True
    


  def __call__(self, data_expanded, data_imputed, sample_b, show_output = False):
    batch = data_imputed
    masks = 1-sample_b
    init_shape = batch.shape[0]


    if torch.cuda.is_available():
        batch = batch.cuda()
        masks = masks.cuda()


    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1
    

  
    with torch.no_grad():
      batch = (batch * 0.3081 + 0.1307) / 255.
      # compute imputation distributions parameters
      samples_params = self.network.generate_samples_params(batch.detach(),
                                                    masks.detach(),
                                                    nb_imputation,
                                                    need_observed = False)
      # img_samples = []
      # for element in samples_params :
      img_samples = self.sampler(samples_params, multiple = True).flatten(0,1)

      img_samples = (img_samples * 255. -  0.1307)/0.3081

    
    _, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    new_data = img_samples.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 
    return new_data, data_expanded, sample_b





class MICE_imputation(MultipleImputation):
  def __init__(self, nb_imputation = 5):
    super().init(nb_imputation = nb_imputation)
    self.network = network


  def __call__(self, data_expanded, data_imputed, sample_b):

    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1
      

    data_expanded_numpy = data_expanded.flatten(1).detach().cpu().numpy()
    mask = sample_b>0.5
    mask = mask.flatten(1).detach().cpu().numpy()
    data_expanded_numpy = np.where(mask, data_expanded_numpy, np.NaN)
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(data_expanded_numpy)
    data_imputed_output = []
    for k in range(nb_imputation):
      data_imputed_output.append(torch.tensor(imp.transform(data_expanded_numpy)).unsqueeze(1))

    data_imputed_output = torch.cat(data_imputed_output, axis=1).flatten(0,1)

    _, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    
    new_data = data_imputed_output.cuda() *  (1-sample_b) + data_expanded * sample_b 
    new_data = new_data

    return new_data, data_expanded, sample_b

  
class MICE_imputation_pretrained(MultipleImputation):
  def __init__(self,network, nb_imputation = 5):
    super().__init__(nb_imputation = nb_imputation)
    self.network = network

  def __call__(self, data_expanded, data_imputed, sample_b):
    
    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1
    


    data_expanded_numpy = data_expanded.flatten(1).detach().cpu().numpy()
    mask = sample_b>0.5
    mask = mask.flatten(1).detach().cpu().numpy()
    data_expanded_numpy = np.where(mask, data_expanded_numpy, np.NaN)
    
    data_imputed_output = []
    for k in range(nb_imputation):
      data_imputed_output.append(torch.tensor(self.network.transform(data_expanded_numpy)).unsqueeze(1))

    data_imputed_output = torch.cat(data_imputed_output, axis=1).flatten(0,1)

    _, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)

    new_data = data_imputed_output.cuda() *  (1-sample_b) + data_expanded * sample_b 
    new_data = new_data

    return new_data, data_expanded, sample_b





class MarkovChain():
  def __init__(self, train_loader, total_init_probability = None, total_transfer_probability = None):
    example_x, target = next(iter(train_loader))
    batch_size, output_dim, sequence_len = example_x.shape
    self.sequence_len = sequence_len
    self.output_dim = output_dim
    if total_transfer_probability is None or total_init_probability is None :
      self.train(train_loader)
    else :
       self.init_probability = total_init_probability
       self.transition_probability = total_transfer_probability

    self.log_init_probability = torch.log(self.init_probability)
    self.log_transition_probability = torch.log(self.transition_probability)

  def train(self, train_loader):
    print("Start training Markov Chain")
    self.init_probability = np.zeros((self.output_dim))
    self.transition_probability = np.zeros((self.output_dim, self.output_dim))
    
    for element, _ in iter(train_loader):
      for sequence in element :
        sequence = sequence.transpose(0,1)
        self.init_probability += sequence[0].numpy()

        for k in range(1,len(sequence)):
          self.transition_probability[torch.argmax(sequence[k-1]).item()] += sequence[k].numpy()

    self.init_probability = torch.tensor(self.init_probability/np.sum(self.init_probability))
    for k in range(self.output_dim):
      self.transition_probability[k] /= np.sum(self.transition_probability[k])
    
    self.transition_probability = torch.tensor(self.transition_probability, dtype = torch.float32)

  def impute(self, data, masks, nb_imputation):
    data_argmax = torch.argmax(data, axis=1)
    for k in range(data_argmax.shape[0]):
      mask = masks[k,0]
      x = data_argmax[k]
      message = torch.zeros((self.sequence_len, self.output_dim))


      # Forward :
      message[0] = self.init_probability # mESSAGE I is arriving at i
      for i in range(1, self.sequence_len):
        if mask[i] == 0:
          message[i] = self.transition_probability[x[i]] # Il faut trouver une façon convenable d'écrire ça avec des batchs
        else :
          message[i] = torch.matmul(message[i-1], self.transition_probability)
        message[i] = message[i]/torch.sum(message[i])
      
      # Backward : 
      output_sample = torch.zeros((nb_imputation,self.sequence_len))
      if mask[self.sequence_len-1]==0 :
        dist = torch.distributions.categorical.Categorical(probs = message[-1]/torch.sum(message[-1],axis=-1))
        aux = dist.sample((torch.tensor(nb_imputation),))
        output_sample[:, -1] = aux
      else :
        output_sample[:, -1] = x[-1].expand(nb_imputation)

      message = message.unsqueeze(1).expand(-1, nb_imputation,-1)
      for i in range(self.sequence_len-2, -1, -1):
        if mask[i] == 0 :
          for l in range(nb_imputation):
            message[i, l, :] *= self.transition_probability[:,output_sample[l,i+1].type(torch.int64)]
            message[i, l, :] = message[i,l,:]/torch.sum(message[i,l])
          dist = torch.distributions.categorical.Categorical(probs=message[i])
          output_sample[:, i] = dist.sample()
        else :
          output_sample[:, i] = x[i].expand(nb_imputation)
            
    # Combine
      if k == 0 :
        output_total = output_sample.unsqueeze(0)
      else :
        output_total = torch.cat([output_total, output_sample.unsqueeze(0)], dim = 0)

    output_total = torch.nn.functional.one_hot(output_total.type(torch.int64),num_classes=self.output_dim).transpose(-1, -2)
    return output_total


# Markov chain imputation

class MarkovChainImputation(MultipleImputation):
  def __init__(self, markov_chain, nb_imputation = 10, to_train = False, use_cuda = False, deepcopy= False):
    super().__init__(nb_imputation=nb_imputation)
    self.markov_chain = markov_chain    


  def __call__(self, data_expanded, data_imputed, sample_b, show_output = False):
    if not self.eval_mode :
      nb_imputation = self.nb_imputation
    else :
      nb_imputation = 1

    with torch.no_grad():
      output = self.markov_chain.impute(data_expanded, sample_b, nb_imputation = self.nb_imputation).cuda()
    
  
    _, data_expanded, sample_b = expand_for_imputations(data_imputed, data_expanded, sample_b, nb_imputation)
    output = output.reshape(data_expanded.shape)
    new_data = output.detach() *  (1-sample_b) + data_expanded.detach() * sample_b 

    return new_data, data_expanded, sample_b