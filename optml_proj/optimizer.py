import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import os
from collections import OrderedDict
from torch.utils.data import Subset
from torch.optim.optimizer import Optimizer
from model_util import *

# Error FeedBack SignSGD
# Ref - https://arxiv.org/pdf/1901.09847.pdf.
# lr is the step-size. 
class EFSGD(Optimizer):
    def __init__(self, params, lr):
        super(EFSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                error_corr = state['error_correction']
                lr = state['lr']
                p = param.grad.data
                p = lr*p + error_corr 
                
                #EFSGD
                g = ( torch.sum( torch.abs(p) )/p.nelement() ) * torch.sign(p)
                state['error_correction'] = p - g
                # The final gradient update to the weights
                state['update'] = g

# Sign SGD - compresses the gradient to its sign.
# lr is the step-size.
class signSGD(Optimizer):
    def __init__(self, params, lr ):
        super(signSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                # The final gradient update to the weights
                state['update'] = lr*param.grad.data.sign()

# QSGD lossy - this does a lossy compression of the gradient
# Ref - https://arxiv.org/abs/1610.02132
# lr is the step-size. 
class QSGD_lossy(Optimizer):
    def __init__(self, params, lr ):
        super(QSGD_lossy,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr
                state['update'] = torch.zeros_like( param.data )
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                # The final gradient update to the weights
                state['update'] = lr*quantizer_lossy(param.grad.data)

# QSGD lossy - this chose the coordinates having top k absolute value 
# the remaining co-ordinates will be set to zeros.
class QSGD_topk(Optimizer):
    def __init__(self, params, lr ):
        super(QSGD_topk,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                # The final gradient update to the weights
                state['update'] = lr*quantizer_topk(param.grad.data)

# QEFSGD lossy - lossy compression with error feedback
# Ref - https://arxiv.org/abs/1806.08054 
# Beta , alpha according to the reference above.  
# the remaining co-ordinates will be set to zeros. 
# lr is the step-size.                        
class QEFSGD_lossy(Optimizer):
    def __init__(self, params, lr,beta,alpha):
        super(QEFSGD_lossy,self).__init__( params , dict( lr = lr, beta = beta,alpha=alpha) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                state['beta'] =beta
                state['alpha'] =alpha
                
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                # The final gradient update to the weights
                state['update'] = lr*quantizer_lossy(state['error_correction']*state['alpha'] + param.grad.data)
                state['error_correction'] = state['beta']*state['error_correction']- state['update'] +param.grad.data 

# topk quantized SGD with error feedback.
# Ref - https://arxiv.org/abs/1806.08054
# Beta , alpha according to the reference above.                  
# lr is the step-size. 
class QEFSGD_topk(Optimizer):
    def __init__(self, params, lr,beta=0.9,alpha=0.1):
        super(QEFSGD_topk,self).__init__( params , dict( lr = lr, beta = beta,alpha=alpha) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['error_correction'] = torch.zeros_like( param.data )
                state['lr'] = lr
                state['beta'] =beta
                state['alpha'] =alpha
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                # The final gradient update to the weights
                state['update'] = lr*quantizer_topk(state['error_correction']*state['alpha'] + param.grad.data)
                state['error_correction'] = state['beta']*state['error_correction'] - state['update'] +param.grad.data 
                

class localSGD(Optimizer):
    def __init__(self, params, lr):
        super(localSGD,self).__init__( params , dict( lr = lr ) )
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['lr'] = lr
                    
    def step(self):
        for group in self.param_groups:
            for k,param in enumerate(group['params']):
                if param.grad is None:
                    continue
                state = self.state[param] 
                lr = state['lr']
                # The final gradient update to the weights
                state['update'] = lr*param.grad.data
