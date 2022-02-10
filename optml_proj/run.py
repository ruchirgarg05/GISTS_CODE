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
from torch.utils.data import Subset, Dataset
from sklearn.datasets import fetch_rcv1
import tqdm
from network import *
from optimizer import *
from model_util import *
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=8,shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

#Loss function
criterion = nn.CrossEntropyLoss()

# Number of workers.
num_workers = 9
sqrt_num_workers = 3
m = trainset_node_split(trainset, num_workers)
trainloaders = [torch.utils.data.DataLoader(m[i], batch_size=32, shuffle=True, num_workers=2) for i in range(num_workers)]


W = torus(sqrt_num_workers)
lrs = []

# uncommit here to run for the ring
# W = ring(num_workers)
# lrs = []



# EFSGD, signSGD, QSGD_lossy, QEFSGD_lossy, QSGD_topk, QEFSGD_topk
# can choose optimizer from one of the above.
optimizer = EFSGD




for i in range(num_workers):
	# Uncomment this in case of QEFSGD_lossy
    # lrs.append({'lr':1e-3,'beta':0.9,'alpha':0.1})

    # rates alter for optimizer - it is documented in the report
    lrs.append({'lr':1e-3})

models = [Net() for i in range(num_workers)]

attacks = ['full_reversal','random_reversal', '' ]
protecs = ['median','trmean','majority',None, 'frac_mean'  ]

# Change the attack and protects from the array above
attack = ''
protec = None

# Byzantine nodes
# Add byzantine nodes to this array
# To run specific experiments add specific nodes - more information on this is available in the report
byzantine_nodes = []

beta_protec = 2/5
# beta_protec = 2/5 ( choose according to architecture )
# Used when running the protect method using frac_mean

net = Network(W, models, m, lrs, trainloaders, 32, nn.CrossEntropyLoss(), device, testloader, signSGD, [3,5], attack, protec, beta_protec  )

iterations = 3000
epochs = 1
results = net.simulate(iterations, epochs)

train_accuracy = [ [ results[i][j]['train_acc'] for j in range(len(a[i])) ] for i in range(9) ]
test_concensus = [ [ results[i][j]['consensus_test'] for j in range(len(b[i])) ] for i in range(9) ]

iterations = [ [ results[i][j]['iteration'] for j in range(len(a[i])) ] for i in range(9) ]

# can alter this for various plots
plot_node = 0

plt.plot(iterations[plot_node],train_accuracy[plot_node],label="train_accuracy")
plt.plot(iterations[plot_node],test_concensus[plot_node],label="test_accuracy")
plt.legend()
plt.title("Consensus Train accuracy for Torus")
plt.ylabel("accuracy")
plt.xlabel("Iterations")
plt.savefig("Consensus_train_accuracy_Torus.jpg")

# Uncomment to dump a pickle file.
# pickle.dump(a,open("ring_signsgd_fullr_2node.pickle","wb"))



