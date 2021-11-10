import torch 
import torch as nn

# custom import
import models.simsiam_builder
from main import pred_dim, dim, momentum, weight_decay, init_lr

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
    model.load_state_dict(torch.load("export.pt"))
    model = model.to(device)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    print("Fine-tuning model...")
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print("Loss: ", loss)
            print("Accuracy: ", acc1, acc5)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
