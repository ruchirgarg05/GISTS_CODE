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


#Majority Vote
#https://arxiv.org/pdf/1810.05291.pdf

def get_vote(grads):
    """Grads is a list of vectors coming from the node and its neighbors only, grads[0] = node.grad"""
    V = torch.zeros_like(grads[0])
    
    for i in grads:
        V+=torch.sign(i.clone().detach())
    
    return V

## Final update on every worker as w = w - neta*(V + lambda*w)(lambda = regularization parameter)




#Robust SGD, Robust One Round
#https://arxiv.org/pdf/1803.01498.pdf


def get_statistic(grads, option = 1, beta = 1/3):
    """option=1 == median, option = 2 == mean"""
    
    # Stack the gradients to perform various statistics
    V = torch.stack(grads, dim=0)
    
    if(option ==1):
        # Take the median along the stacked dimension.
        values, indices = torch.median(V, dim=0)
        temp = values.clone().detach()
    else:
        # Sort the coordinates to take in th fraction of [ beta, 1 - beta ].
        m = torch.sort(V, dim=0)[0].clone().detach()
        first_index = int(beta*m.size()[0])
        last_index = int((1-beta)*m.size()[0])
        
        total = last_index - first_index
        
        temp = torch.zeros_like(grads[0])
        
        if(total > 0):
            for i in range(total):
                temp+=m[i+first_index]

            temp = temp/total
    
    return temp.clone().detach()


# Sort across norm of gradients and take mean excluding the largest beta fraction.
def get_frac(grads, beta = 1/3):
    V = torch.tensor([torch.norm(grad) for grad in grads])
    _,idx = torch.sort(V)
    t = int((1-beta)*len(grads))
    id_keep = idx[:t].int().tolist()
    new_grads = [grads[i].clone() for i in id_keep]
    
    return sum(new_grads)/len(new_grads)
