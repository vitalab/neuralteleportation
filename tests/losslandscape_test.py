import torch
import torch.nn as nn
import torchvision
import numpy as np

import copy
import argparse

from neuralteleportation.losslandscape.plot_surface import SurfacePlotter

from neuralteleportation.utils.dataloader import load_cifar10_dataset
import neuralteleportation.losslandscape.net_plotter as net_plotter
import neuralteleportation.losslandscape.plot_2D as plot_2D


class Namespace:
    '''
    siple namespace that imitate the argparser namespace.
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    batch_size = 128
    raw_data = True
    threads = 1
    data_split = 1
    split_idx = 0
    trainloader = ''
    testloader = ''
    model = ''
    model_folder = ''
    model_file = ''
    model_file2 = ''
    model_file3 = ''
    dir_file = ''
    dir_type = 'weights'
    surf_file = ''
    x = '-1:1:51'
    xnorm = 'filter'
    xignore = 'biasbn'
    y = '-1:1:51'
    ynorm = 'filter'
    yignore = 'biasbn'
    same_dir = False
    idx = 0

    trainloader, testloader = load_cifar10_dataset(batch_size, raw_data, data_split, split_idx)

    # simple model to create landscape from.
    net = nn.Sequential(nn.Conv2d(3, 64, 3, bias=False),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(57600, 10, bias=False))
    weights = net_plotter.get_weights(net)
    state = copy.deepcopy(net.state_dict)

    surfplt = SurfacePlotter(net, device, x, y, surf_file)

    criterion = nn.CrossEntropyLoss()
    surfplt.crunch(net, criterion, weights, state, trainloader, 'train_loss', 'train_acc')

    vmin = 0.1
    vmax = 10
    vlevel = 0.5
    loss_max = 5
    log = True

    surfplt.plot_surface(vmin, vmax, vlevel, loss_max, log)
