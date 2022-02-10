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

criterion = nn.CrossEntropyLoss()

num_workers = 10

models_full = [Net() for i in range(num_workers)]
models_ring = [Net() for i in range(num_workers)]
learning_rate = 1e-3 * torch.ones(num_workers)

m = trainset_node_split(trainset, num_workers)
trainloaders = [torch.utils.data.DataLoader(m[i], batch_size=4, shuffle=True, num_workers=2) for i in range(num_workers)]

full_connected = torch.ones([num_workers,num_workers])
net_full = Network(full_connected,models_full, 1e-3 * torch.ones(num_workers),trainloaders, nn.CrossEntropyLoss())

ring = torch.zeros([num_workers, num_workers])
for i in range(num_workers-1):
    ring[i,i+1] = 1.0
    ring[i,i-1] = 1.0
#close
ring[num_workers - 1, 0 ] = 1
ring[num_workers - 1, num_workers-2 ] = 1
net_ring = Network(full_connected,models_ring, 1e-3 * torch.ones(num_workers),trainloaders, nn.CrossEntropyLoss())

num_iterators = 5*1e4

loss_complete = []
for i in range( int( num_iterators/500) ):
    net_full.simulate( 500 , 1 )
    _,loss = forward_test( models_full[0], trainloader )
    loss_complete.append(loss)

loss_ring = []
for i in range( int( num_iterators/500) ):
    net_ring.simulate( 500 , 1 )
    _,loss = forward_test( models_ring[0], trainloader )
    loss_ring.append(loss)


# Gradient_reversal attack
criterion = nn.CrossEntropyLoss()

num_workers = 10

# with out loss of generality make the last two nodes errorneous. Simulate this by just changing the sign of the gradient.

models_full_byz = [Net() for i in range(num_workers)]
models_ring_byz = [Net() for i in range(num_workers)]
learning_rate = 1e-3 * torch.ones(num_workers)

m = trainset_node_split(trainset, num_workers)
trainloaders = [torch.utils.data.DataLoader(m[i], batch_size=4, shuffle=True, num_workers=2) for i in range(num_workers)]

full_connected_byz = torch.ones([num_workers,num_workers])
for i in range(num_workers):
  full_connected_byz[i,num_workers - 1 ] = -1
  full_connected_byz[i,num_workers - 2 ] = -1
net_full_byz = Network(full_connected_byz,models_full_byz, 1e-3 * torch.ones(num_workers),trainloaders, nn.CrossEntropyLoss())

ring_byz = torch.zeros([num_workers, num_workers])
for i in range(num_workers-1):
    ring_byz[i,i+1] = 1.0
    ring_byz[i,i-1] = 1.0
#close
ring_byz[num_workers - 1, 0 ] = 1
ring_byz[num_workers - 1, num_workers-2 ] = 1
faulty = [ int( num_workers/2 ), num_workers-1 ]
for i in range(num_workers):
  for j in faulty:
    ring_byz[i,j] = ring_byz[i,j]*-1
net_ring_byz = Network(full_connected_byz,models_ring_byz, 1e-3 * torch.ones(num_workers),trainloaders, nn.CrossEntropyLoss())

num_iterators = 5*1e4

loss_complete_byz = []
for i in range( int( num_iterators/500) ):
    net_full_byz.simulate( 500 , 1 )
    _,loss = forward_test( models_full_byz[0], trainloader )
    loss_complete_byz.append(loss)

loss_ring_byz = []
for i in range( int( num_iterators/500) ):
    net_ring_byz.simulate( 500 , 1 )
    _,loss = forward_test( models_ring_byz[0], trainloader )
    loss_ring_byz.append(loss)