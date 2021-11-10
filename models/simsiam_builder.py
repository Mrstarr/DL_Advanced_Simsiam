import torch
import torch.nn as nn
import torchvision.models as models

"""
SimSiam model
"""
class SimSiam(nn.Module):

    def __init__(self, dim=512, pred_dim=128):
        """
        dim: feature dimension (default: 2048), The hidden fc is 2048-d
        pred_dim: hidden dimension of the predictor, according to paper = 512
        """

        super(SimSiam, self).__init__()

        # use ResNet18 as backbone
        backbone = models.__dict__['resnet18']
        self.encoder = backbone(num_classes=dim, zero_init_residual=True)

        # build the 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build the 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            Algorithm 1 in https://arxiv.org/abs/2011.10566
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach() # stop-gradient

    def forward_single(self, x):
        z = self.encoder(x)
        return z

"""
SimSiam model
"""
class SimSiamWrapper(nn.Module):

    def __init__(self, simsiam_model, dim, freeze=True, num_classes=10):
        """
        dim: feature dimension (default: 2048), The hidden fc is 2048-d
        pred_dim: hidden dimension of the predictor, according to paper = 512
        """

        super(SimSiamWrapper, self).__init__()

        # use ResNet18 as backbone
        self.encoder = simsiam_model.encoder
        if freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        self.fc = nn.Linear(dim, num_classes)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()


    def forward(self, x):
        z = self.encoder(x)
        return self.fc(z)