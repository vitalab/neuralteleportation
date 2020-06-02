"""
"""
import torch
import torchvision
from torchvision import transforms
import os
import numpy as np
import argparse


def load_cifar10_dataloaders(batch_size=128, data_split=1, split_idx=0, shuffle=False, download=True):
    """
    Setup dataloader. The data is not randomly cropped as in training because of
    we want to esimate the loss value with a fixed dataset.

    Args:
        data_split: the number of splits for the training dataloader
        split_idx: the index for the split of the dataloader, starting at 0

    Returns:
        train_loader, test_loader
    """

    assert split_idx < data_split, 'the index of data partition should be smaller than the total number of split'
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='/tmp', train=True, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset = torchvision.datasets.CIFAR10(root='/tmp', train=False, download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--dataset', default='cifar10', help='cifar10')
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--shuffle', default=False, type=bool, help='If the DataLoader should be shuffle or not.')

    args = parser.parse_args()

    trainloader, testloader = load_cifar10_dataloaders(args.batch_size, args.raw_data, args.data_split, args.split_idx, args.shuffle)

    print('num of batches: %d' % len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('batch_idx: %d   batch_size: %d'%(batch_idx, len(inputs)))
