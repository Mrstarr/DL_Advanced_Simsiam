import torch
import torch.nn as nn

# custom import
import models.simsiam_builder

batch_size = 16
lr = 0.1
gpu = None


'''
Create and train a simsiam model
'''
print("Creating SimSiam model...")
simsiam =  models.simsiam_builder.SimSiam()

# infer learning rate before changing batch size
init_lr = lr * batch_size / 256
# define loss function (criterion) and optimizer
criterion = nn.CosineSimilarity(dim=1).cuda(gpu)