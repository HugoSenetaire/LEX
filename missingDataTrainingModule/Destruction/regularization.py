import torch




def free_regularization(pi_list):
  loss_reg = torch.mean(torch.mean(pi_list,-1)).squeeze() 
  return loss_reg

def squared_regularization(pi_list, missing_rate = 0.5):
  batch_size = pi_list.shape[0]
  channels = pi_list.shape[1]
  regularizing_vector = torch.tensor([missing_rate])[:,None].expand(batch_size,channels).cuda()
  loss_reg =  torch.mean((torch.mean(pi_list,-1) - regularizing_vector)**2).squeeze() # Not absolute or squared ? Intger or rate ?
  return loss_reg