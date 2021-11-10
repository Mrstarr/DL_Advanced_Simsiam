import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle

import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_train = 128
batch_size_test = 2048

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def load_mnist(augment_images=True):
    normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))

    augmentation = [
        transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, (.1, 2.))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    transform = TwoCropsTransform(transforms.Compose(augmentation))

    normalize = transforms.Compose([transforms.ToTensor(), normalize])

    if augment_images:
        dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                             transform=transform)
    else:
        dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                             transform=normalize)


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

    dataset_test = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                              transform=normalize)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)


    return train_loader, test_loader


def load_cifar(augment_images=True):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, (.1, 2.))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    transform = TwoCropsTransform(transforms.Compose(augmentation))

    normalize = transforms.Compose([transforms.ToTensor(), normalize])

    if augment_images:
        dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                               transform=transform)
    else:
        dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                               transform=normalize)


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

    dataset_test = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                                transform=normalize)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

# Example with MNIST
if __name__ == '__main__':
    train, test = load_mnist()

    for (x1, x2), labels in train:
        x1 = x1[0]
        plt.imshow(  x1.permute(1, 2, 0).reshape(28, 28)  )
        plt.show()
        x2 = x2[0]
        plt.imshow(  x2.permute(1, 2, 0).reshape(28, 28)  )
        plt.show()
        break
