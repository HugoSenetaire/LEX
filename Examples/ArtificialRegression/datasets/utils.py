import torch

def f_a(X):
    return X[:,0]*X[:,1]

def getProbA(X):
    fa = f_a(X)
    aux_fa = torch.exp(fa)
    b_fa = 1/(1+aux_fa)
    sel = torch.zeros_like(X)
    sel[:,:2] = 1
    return fa, b_fa, sel

def f_b(X):
    return X[:,2:6].pow(2).sum(axis = 1) - 4

def getProbB(X):
    fb = f_b(X)
    aux_fb = torch.exp(fb)
    b_fb = 1/(1+aux_fb)
    sel = torch.zeros_like(X)
    sel[:,2:6] = 1
    return fb, b_fb, sel 

def f_c(X):
    return -10*torch.sin(0.2*X[:,6]) + torch.abs(X[:,7]) + X[:,8] + torch.exp(-X[:,9])-2.4

def getProbC(X):
    fc = f_c(X)
    aux_fc = torch.exp(fc)
    b_fc = 1/(1+aux_fc)
    sel = torch.zeros_like(X)
    sel[:,6:10] = 1
    return fc, b_fc, sel



def f_prod(X, used_dim):
    f = torch.prod(X[:,:used_dim], axis = 1)
    aux_f = torch.exp(f)
    b_f = 1/(1+aux_f)
    sel = torch.zeros_like(X)
    sel[:,:used_dim] = 1

    return f, b_f, sel

def f_squaredsum(X, used_dim):
    f = torch.sum(X[:,:used_dim]**2, axis = 1) - 4
    aux_f = torch.exp(f)
    b_f = 1/(1+aux_f)
    sel = torch.zeros_like(X)
    sel[:,:used_dim] = 1

    return f, b_f, sel

def f_squaredsum2(X, used_dim):
    f = torch.sum(X[:,:used_dim]**2, axis = 1) - used_dim
    aux_f = torch.exp(f)
    b_f = 1/(1+aux_f)
    sel = torch.zeros_like(X)
    sel[:,:used_dim] = 1
    return f, b_f, sel
