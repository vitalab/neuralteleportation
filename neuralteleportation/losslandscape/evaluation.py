"""
    Authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
    Title: Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.
    Source Code: https://github.com/tomgoldstein/loss-landscape

    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix

    Modified: Philippe Spino
    Last Modified: 1 June 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable


def eval_loss(net, criterion, loader, device='cpu'):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        device: torch.device cpu or cuda.
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0  # number of samples

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for _, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()
        elif isinstance(criterion, nn.MSELoss):
            for _, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()
    return total_loss / total, 100. * correct / total
