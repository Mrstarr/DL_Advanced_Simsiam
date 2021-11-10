import torch
import math

'''
From https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
'''
def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

'''
Function to train SimSiam net
'''
def train_simsiam(train_loader, model, criterion, optimizer, epoch, out_dim, device):
    # Set model in training mode, if not already
    model.train()
    data_loader_size = len(train_loader)
    avg_loss = 0
    avg_output_std = 0
    for i, ((x1, x2), _) in enumerate(train_loader):
        print(f"{i}/{data_loader_size}", end="\r")

        x1 = x1.to(device)#cuda(gpu, non_blocking=True)
        x2 = x2.to(device)#cuda(gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=x1, x2=x2)

        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        z1 = torch.nn.functional.normalize(z1.detach(), dim=1)

        output_std = torch.std(z1, 0)
        output_std = output_std.mean()

        w = 0.9
        avg_loss = w * avg_loss + (1-w)*loss.item()
        avg_output_std = w* avg_output_std + (1-w)*output_std.item()

    collapse_level = max(0, 1 - math.sqrt(out_dim) * avg_output_std)

    print(f'[Epoch:{epoch:3d}]'
        f'Loss={avg_loss:.2f}|'
        f'Collapse Level:{collapse_level:.2f}'
    )