import torch
import torch.nn as nn
import argparse

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.losslandscape import net_plotter
from neuralteleportation.training.experiment_setup import get_cifar10_datasets, get_mnist_datasets
from neuralteleportation.training.training import train
from neuralteleportation.training.config import TrainingMetrics, TrainingConfig
from neuralteleportation.metrics import accuracy
from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.losslandscape.surfaceplotter import SurfacePlotter


def argumentparser():
    parser = argparse.ArgumentParser(description="Simple argument parser for the 1D-interpolation-plot experiment")
    parser.add_argument("--cuda", "-c", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--x", type=str, default="-1:1:10")
    parser.add_argument("--xignore", type=str, default="")

    return parser.parse_args()


if __name__ == '__main__':
    args = argumentparser()

    device = 'cpu'
    if torch.cuda.is_available() and args.cuda:
        device = 'cuda'

    trainset, valset, testset = get_cifar10_datasets()
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=args.batch_size)

    net = resnet18COB(num_classes=10, input_channels=3).to(device=device)
    net = NeuralTeleportationModel(net, (32, 3, 32, 32))

    metric = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])
    config = TrainingConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=device)
    train(net, trainset, metric, config)

    w = net_plotter.get_weights(net)

    srfplt = SurfacePlotter(net_name='test', net=net, x=args.x, xignore=args.xignore)
    srfplt.crunch(nn.CrossEntropyLoss(),
                  w, None,
                  dataloader=trainloader,
                  loss_key='train_loss', acc_key='train_acc',
                  device=device)
    srfplt.plot_1d_loss()
