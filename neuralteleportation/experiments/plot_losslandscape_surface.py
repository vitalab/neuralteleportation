import torch
import torch.nn as nn
import argparse
from itertools import chain

from neuralteleportation.losslandscape.surfaceplotter import SurfacePlotter
from neuralteleportation.losslandscape import net_plotter as net_plotter

from neuralteleportation.models.model_zoo import resnetcob, densenetcob, vggcob
from neuralteleportation.models.model_zoo.resnetcob import __all__ as __resnets__
from neuralteleportation.models.model_zoo.densenetcob import __all__ as __densenets__
from neuralteleportation.models.model_zoo.vggcob import __all__ as __vggnets__
from neuralteleportation.training import experiment_setup
from neuralteleportation.training.training import train
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


# Creating a list of all the possible models.
__models__ = list(chain.from_iterable([__resnets__, __densenets__, __vggnets__]))


def argument_parser():
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the plot loss landscape experiment.')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10,
                        help='How many epoch to train the network if train set to true')
    parser.add_argument("--cuda", "-c", action="store_true", default=False, help='use cuda if it is availble')
    parser.add_argument("--load_model", type=str, default="", help="file path of the h5 network state file.")
    parser.add_argument("--train", "-t", action="store_true", default=False,
                        help="if the model should be train before teleportation")
    parser.add_argument("--save_model", "-s", action="store_true", default=False, help="tell to save the model or not")
    parser.add_argument("--save_model_location", type=str, default='/tmp/model.pt',
                        help="save path and .pt file forthe selected network")
    parser.add_argument("--cob_range", type=float, default=0.5, help='set the CoB range for the teleportation.')
    parser.add_argument("--teleport_mode", type=str, default="positive", choices=["positive", "negative"],
                        help="set the teleportation mode between positive and negative CoB")
    parser.add_argument("--sampling", type=str, default="usual", help="Sampling type for CoB.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"],
                        help="what dataset should the model be trained on.")
    parser.add_argument("--x", type=str, default="-1:1:5", help='x axis size and precision')
    parser.add_argument("--y", type=str, default="-1:1:5", help='y axis size and precision')
    parser.add_argument("--xignore", type=str, default="", help='ignore biasbn')
    parser.add_argument("--yignore", type=str, default="", help='ignore biasbn')
    parser.add_argument("--same_direction", action="store_true", default=False,
                        help="Should the SurfacePlotter use the same direction for the y axis.")
    parser.add_argument("--model", "-m", type=str, default="vgg11COB", choices=__models__)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate used for the training if train is active.")
    return parser.parse_args()


def get_model_from_string(args):
    if args.model in __resnets__:
        func = getattr(resnetcob, args.model)
    elif args.model in __densenets__:
        func = getattr(densenetcob, args.model)
    elif args.model in __vggnets__:
        func = getattr(vggcob, args.model)
    else:
        raise Exception("%s was not found in the model zoo" % args.model)
    model = func(pretrained=False, num_classes=10)
    return model


if __name__ == "__main__":
    args = argument_parser()

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.cuda:
        use_cuda = True
        device = torch.device('cuda')

    if args.dataset == "cifar10":
        trainset, valset, testset = experiment_setup.get_cifar10_datasets()
    elif args.dataset == "mnist":
        trainset, valset, testset = experiment_setup.get_mnist_datasets()

    data_size = (args.batch_size, trainset.data.shape[3], trainset.data.shape[1], trainset.data.shape[2])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

    if args.load_model:
        net = torch.load(args.load_model, map_location=device)
        net = NeuralTeleportationModel(net, input_shape=data_size).to(device)
    else:
        net = NeuralTeleportationModel(get_model_from_string(args), input_shape=data_size).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.train:
        metric = TrainingMetrics(criterion, [accuracy])
        config = TrainingConfig(lr=args.lr, epochs=args.epochs, device=device, batch_size=args.batch_size)
        train(net, train_dataset=trainset, metrics=metric, config=config)

    if args.save_model:
        torch.save(net, args.save_model_location)

    if args.teleport_mode == "positive":
        net = net.random_teleport(cob_range=args.cob_range, sampling_type=args.sampling)
    elif args.teleport_mode == "negative":
        net = net.random_teleport(cob_range=-args.cob_range, sampling_type=args.sampling)
    else:
        raise NotImplementedError("%s mode not yet implemented or mode doesn't exist" % args.teleport_mode)

    w = net_plotter.get_weights(net)

    surfplt = SurfacePlotter(args.model, net, args.x, args.y, same_direction=args.same_direction,
                             xignore=args.xignore, yignore=args.yignore)
    surfplt.crunch(criterion, w, None, trainloader, 'train_loss', 'train_acc', device)
    surfplt.plot_surface()
    surfplt.show()
