import torch
import torch.nn as nn

# custom import
import models.simsiam_builder
from train import train_simsiam, adjust_learning_rate
from dataloader import load_cifar

# Hyperparams, added defualt values
batch_size = 16
lr = 0.1
gpu = None
momentum = 0.9
weight_decay = 1e-4
start_epoch = 0
epochs = 100


'''
Load data
'''
print("Loading CIFAR...")
train_loader, test_loader = load_cifar()


'''
Create and train a simsiam model
'''
print("Creating SimSiam model...")
simsiam =  models.simsiam_builder.SimSiam()
# infer learning rate before changing batch size
init_lr = lr * batch_size / 256
# define loss function (criterion) and optimizer
criterion = nn.CosineSimilarity(dim=1).cuda(gpu)
optim_params = simsiam.parameters()
optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, epochs)
        train_simsiam(train_loader, simsiam, criterion, optimizer, epoch, gpu)