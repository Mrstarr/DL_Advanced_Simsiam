import torch
import torch.nn as nn
import numpy as np

# custom import
import models.simsiam_builder
from train import train_simsiam, adjust_learning_rate
from dataloader import load_cifar
from get_latent_space import get_and_save_latents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams, added defualt values
batch_size = 16
lr = 0.05
gpu = None
momentum = 0.9
weight_decay = 1e-4
start_epoch = 0
epochs = 5


'''
Load data
'''
print("Loading CIFAR...")
train_loader, test_loader = load_cifar()


'''
Create and train a simsiam model
'''
print("Creating SimSiam model...")
pred_dim = 64
dim = 128
simsiam =  models.simsiam_builder.SimSiam(dim=dim, pred_dim=pred_dim).to(device)
# infer learning rate before changing batch size
init_lr = lr * batch_size / 256
# define loss function (criterion) and optimizer
criterion = nn.CosineSimilarity(dim=1).to(device)#.cuda(gpu)
optim_params = simsiam.parameters()
optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

model_parameters = filter(lambda p: p.requires_grad, simsiam.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

for epoch in range(start_epoch, epochs):
    if epoch % 10 == 0:
        print("Saving model!")
        torch.save(simsiam.state_dict(), "models/export.pt")

    print("Current Epoch", epoch)
    adjust_learning_rate(optimizer, init_lr, epoch, epochs)
    train_simsiam(train_loader, simsiam, criterion, optimizer, epoch, pred_dim, device)

torch.save(simsiam.state_dict(), "models/export.pt")

# Model can be loaded with:
# model = models.simsiam_builder.SimSiam(dim=dim, pred_dim=pred_dim)
# model.load_state_dict(torch.load("models/export.pt"))
# model = model.to(device)

get_and_save_latents(test_loader, simsiam, device)