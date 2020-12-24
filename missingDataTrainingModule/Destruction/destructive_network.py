import torch.nn as nn
import torch.nn.functional as F
import torch

# Define the generator
class Destructor(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,200)
        self.pi = nn.Linear(200, input_size)

    def __call__(self, x):
        # print(x.shape)
        x = x.flatten(1) # Batch_size, Channels* SizeProduct
        pi = F.elu(self.fc1(x))
        return F.sigmoid(self.pi(pi))

class DestructorVariational(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size+output_size,200)
    self.pi = nn.Linear(200, input_size)

  def __call__(self, x, y):
    # X et Y doivent avoir N_expectation ?
    x = x.flatten(1)  #Batch_size, Channels* SizeProduct
    # y = torch.double(y)
    y = y.float()
 # Y : Batch_size, Ncategory
    x = torch.cat([x,y],1)
    pi = F.elu(self.fc1(x))
    return F.sigmoid(self.pi(pi))
