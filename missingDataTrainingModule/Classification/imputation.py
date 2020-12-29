
import torch


class Imputation():
  def __init__(self, isRounded = False):
    self.isRounded = isRounded
    return True

  def round_sample(self, sample_b):
    if self.isRounded :
      sample_b_rounded = torch.round(sample_b)
      return sample_b_rounded #N_expectation, batch_size, channels, size:...
    else :
      return sample_b

    
  def impute(self,data_expanded, sample_b):
    raise NotImplementedError

  def is_learnable(self):
    return False




class ConstantImputation(Imputation):
  def __init__(self, cste = 0, isRounded = False):
    self.cste = cste
    super().__init__(isRounded = isRounded)
    
  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    return data_expanded * ((1-sample_b) * self.cste + sample_b)



class MaskConstantImputation(Imputation):
  def __init__(self, cste = 0, isRounded = False):
    self.cste = cste
    super().__init__(isRounded = isRounded)

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    return torch.cat([data_expanded * ((1-sample_b) * self.cste + sample_b), sample_b], axis = 2)


class LearnImputation(Imputation):
  def __init__(self, isRounded = False):
    self.learned_cste = torch.zeros((1), requires_grad=True)
    super().__init__(isRounded = isRounded)

  def get_learnable_parameter(self):
    return self.learned_cste

  def zero_grad(self):
    if self.learned_cste.grad is not None :
      self.learned_cste.grad.zero_()

  def impute(self, data_expanded, sample_b):
    sample_b = self.round_sample(sample_b)
    return data_expanded  * ((1-sample_b) * self.learned_cste + sample_b)

  def is_learnable(self):
    return True

