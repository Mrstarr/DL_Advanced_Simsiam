import torch
from torch import nn

# custom import
import models.simsiam_builder
from main import pred_dim, dim, momentum, weight_decay, init_lr
from dataloader import load_cifar

'''
Script which takes a pre-trained model and fine-tunes it for classification
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Computes the accuracy over the k top predictions for the specified values of k
https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py
'''
def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

'''
Fine tune the SimSiam model
'''
def fine_tune_simsiam(train_loader):
    # Import pre-trained model
    print("Loading pre-trained model...")
    model = models.simsiam_builder.SimSiam(dim=dim, pred_dim=pred_dim)
    model.load_state_dict(torch.load("models/export.pt"))
    model = models.simsiam_builder.SimSiamWrapper(model, dim, num_classes=10)
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # optimize only the linear classifier
    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    print("Fine-tuning model...")
    # model.eval()
    # with torch.no_grad():
    model.train()
    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # print accuracy and loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        print("Loss: ", loss.item())
        print("Accuracy: ", acc1.item(), acc5.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def inference_simsiam(test_loader, fine_tuned_model):
    fine_tuned_model.eval()
    # define loss function 
    criterion = nn.CrossEntropyLoss().to(device)

    print("Testing the model...")
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = fine_tuned_model(images)
            #loss = criterion(output, target)

            # print accuracy and loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #print("Loss: ", loss.item())
            print("Accuracy: ", acc1.item(), acc5.item())
            


if __name__ == '__main__':
    train_loader, test_loader = load_cifar(augment_images=False)
    fine_tuned_model = fine_tune_simsiam(train_loader)
    inference_simsiam(test_loader, fine_tuned_model)