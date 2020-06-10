import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.training.experiment_setup import get_model_list, get_model_from_string
from neuralteleportation.training.experiment_setup import get_cifar10_datasets, get_mnist_datasets
from neuralteleportation.training.training import train_epoch, test


__models__ = get_model_list()


def argumentparser():
    parser = argparse.ArgumentParser(description="Simple argument parser for traininng teleportation")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", type=str, default="resnet18COB", choices=__models__)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--teleport_every", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--cob_range", type=float, default=0.5)
    parser.add_argument("--cob_sampling", type=str, default="usual",
                        choices=["usual", "symmetric", "negative", "zero"])

    return parser.parse_args()


if __name__ == '__main__':
    args = argumentparser()

    device = 'cpu'
    if torch.cuda.is_available() and args.cuda:
        device = 'cuda'

    if args.dataset == 'cifar10':
        trainset, valset, testset = get_cifar10_datasets()
    elif args.dataset == 'mnist':
        trainset, valset, testset = get_mnist_datasets()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)

    w, h = trainset.data.shape[1:3]
    dims = 1 if trainset.data.ndim < 4 else trainset.data.shape[3]

    net_vanilla = get_model_from_string(args.model, num_classes=10, input_channels=dims).to(device)
    net_5050 = get_model_from_string(args.model, num_classes=10, input_channels=dims)
    net_5050 = NeuralTeleportationModel(network=net_5050,
                                        input_shape=(args.batch_size, dims, w, h)).to(device).to(device)
    net_teleport = get_model_from_string(args.model, num_classes=10, input_channels=dims)
    net_teleport = NeuralTeleportationModel(network=net_teleport,
                                            input_shape=(args.batch_size, dims, w, h)).to(device).to(device)

    metric = TrainingMetrics(criterion=nn.CrossEntropyLoss(), metrics=[accuracy])
    config = TrainingConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=device)

    optim_vanilla = torch.optim.SGD(lr=args.lr, params=net_vanilla.parameters())
    optim_5050 = torch.optim.SGD(lr=args.lr, params=net_5050.parameters())
    optim_teleport = torch.optim.SGD(lr=args.lr, params=net_teleport.parameters())

    res_vanilla = []
    res_5050 = []
    res_teleport = []

    print("===============================")
    print("========Training Vanilla=======")
    print("===============================")
    for e in np.arange(1, args.epochs + 1):
        train_epoch(model=net_vanilla,
                    criterion=metric.criterion,
                    optimizer=optim_vanilla,
                    train_loader=trainloader,
                    epoch=e,
                    device=device)
        val_res = test(model=net_vanilla, dataset=valset, metrics=metric, config=config)
        res_vanilla.append(val_res['accuracy'])

    print()
    print("===============================")
    print("========Training 5050=======")
    print("===============================")
    b = Bernoulli(torch.tensor([0.5]))
    for e in np.arange(1, args.epochs + 1):
        if e % args.teleport_every == 0 and b.sample() == 1:
            print("teleported model")
            net_5050.random_teleport()
        train_epoch(model=net_5050,
                    criterion=metric.criterion,
                    optimizer=optim_5050,
                    train_loader=trainloader,
                    epoch=e,
                    device=device)
        val_res = test(model=net_5050, dataset=valset, metrics=metric, config=config)
        res_5050.append(val_res['accuracy'])

    print()
    print("===============================")
    print("=======Training Every 4========")
    print("===============================")
    for e in np.arange(1, args.epochs + 1):
        if e % args.teleport_every == 0:
            print("teleported model")
            net_teleport.random_teleport()
        train_epoch(model=net_teleport,
                    criterion=metric.criterion,
                    optimizer=optim_teleport,
                    train_loader=trainloader,
                    epoch=e,
                    device=device)
        val_res = test(model=net_teleport, dataset=valset, metrics=metric, config=config)
        res_teleport.append(val_res['accuracy'])

    plt.figure()
    plt.title("SGD %s epochs training, %s, %s" % (args.epochs, args.model, args.dataset))
    plt.plot(res_vanilla, '-o', label="vanilla", markersize=2)
    plt.plot(res_5050, '-o', label="net_5050", markersize=2)
    plt.plot(res_teleport, '-o', label="net_teleport", markersize=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
