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
def fine_tune_simsiam(train_loader, freeze):
    # Import pre-trained model
    print("Loading pre-trained model...")
    model = models.simsiam_builder.SimSiam(dim=dim, pred_dim=pred_dim)
    model.load_state_dict(torch.load("models/export.pt"))
    model = models.simsiam_builder.SimSiamWrapper(model, dim, freeze=freeze, num_classes=10)
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
    for epoch in range(3):
        acc1_avg = 0
        acc5_avg = 0
        loss_avg = 0
        for i, (images, target) in enumerate(train_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_avg += loss.item()
            acc1_avg += acc1.item()
            acc5_avg += acc5.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Loss: ", loss_avg/len(train_loader))
        print("Accuracy: ", acc1_avg/len(train_loader), acc5_avg/len(train_loader))

    return model

def inference_simsiam(test_loader, fine_tuned_model):
    fine_tuned_model.eval()

    print("Testing the model...")
    acc1_avg = 0
    acc5_avg = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = fine_tuned_model(images)
            #loss = criterion(output, target)

            # print accuracy and loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_avg += acc1.item()
            acc5_avg += acc5.item()
        print("Accuracy: ", acc1_avg/len(test_loader), acc5_avg//len(test_loader))

if __name__ == '__main__':
    train_loader, test_loader = load_cifar(augment_images=False)
    fine_tuned_model = fine_tune_simsiam(train_loader, freeze=True)
    inference_simsiam(test_loader, fine_tuned_model)

    torch.save(fine_tuned_model.state_dict(), "models/fine_tuned_model.pt")
