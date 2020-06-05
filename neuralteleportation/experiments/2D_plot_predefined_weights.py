import torch
import argparse

from torch.utils.data.dataloader import DataLoader

from neuralteleportation.training import experiment_setup
from neuralteleportation.training.training import train
from neuralteleportation.metrics import accuracy
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.losslandscape.surfaceplotter import SurfacePlotter
from neuralteleportation.losslandscape import net_plotter


def argument_parser():
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_teleportation", type=int, default=1,
                        help="the number of time the network should be teleported.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--x", type=str, default="-1:1:10", help='x axis size and precision')
    parser.add_argument("--y", type=str, default="-1:1:10", help='y axis size and precision')
    parser.add_argument("--xignore", type=str, default="", help='ignore biasbn')
    parser.add_argument("--yignore", type=str, default="", help='ignore biasbn')
    parser.add_argument("--same_direction", action="store_true", default=False,
                        help="Should the SurfacePlotter use the same direction for the y axis.")
    # parser.add_argument("")
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = 'cpu'
    if args.cuda:
        device = 'cuda'

    if args.dataset == "mnist":
        trainset, valset, testset = experiment_setup.get_mnist_datasets()
    elif args.dataset == "cifar10":
        trainset, valset, testset = experiment_setup.get_cifar10_datasets()
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

    input_shape = trainset.data.shape[1:]

    net = MLPCOB(num_classes=10, input_shape=input_shape, hidden_layers=[30])
    criterion = torch.nn.CrossEntropyLoss()
    metrics = TrainingMetrics(criterion, [accuracy])
    config = TrainingConfig(lr=args.lr,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            device=device)
    train(net, trainset, metrics=metrics, config=config)
    w = net_plotter.get_weights(net)

    surfplt = SurfacePlotter('MLP', net, x=args.x, y=args.y, xignore=args.xignore, yignore=args.yignore)
    surfplt.crunch(criterion, w, None, dataloader=trainloader, loss_key='train_loss', acc_key='train_acc',
                   device=device)
    surfplt.plot_contours(vmin=0.1, vmax=10, vlevel=0.5)
    surfplt.show()
