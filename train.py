

def train_simsiam(train_loader, model, criterion, optimizer, epoch, gpu):
    for i, (images, _) in enumerate(train_loader):

        if gpu is not None:
            images[0] = images[0].cuda(gpu, non_blocking=True)
            images[1] = images[1].cuda(gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()