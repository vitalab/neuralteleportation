import pathlib
import argparse

import torch
import torch.nn as nn

from neuralteleportation.training.experiment_setup import get_dataset_subsets, get_model, get_model_names
from neuralteleportation.training.training import train, test
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.metrics import accuracy
from neuralteleportation.losslandscape import generate_random_2d_vector, generate_contour_loss_values, plot_contours

from neuralteleportation.losslandscape import contour_checkpoint_file as checkpoint_file
from neuralteleportation.utils.pathutils import get_nonexistent_path


def argument_parser():
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the plot loss landscape experiment.')

    # Hyper Parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10,
                        help='How many epoch to train the network if train set to true')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate used for the training if train is active.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="Select the Gradient descent optimizer for the experiment")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"],
                        help="Selects the dataset use in the experiment")
    parser.add_argument("--cob_range", type=float, default=0.5, help='set the CoB range for the teleportation.')
    parser.add_argument("--cob_sampling", type=str, default="intra_landscape", help="Sampling type for CoB.")

    # Experiment Configuration
    parser.add_argument("--train", "-t", action="store_true", default=False,
                        help="if the model should be train before teleportation")
    parser.add_argument("--x", nargs=3, type=float, default=[-1, 1, 41],
                        help="Defines the precision of the x")
    parser.add_argument("--y", nargs=3, type=float, default=[-1, 1, 41],
                        help="Defines the precision of the y")
    parser.add_argument("--use_bias_bn", action="store_true", default=False,
                        help="Whether or not to consider bias in layer and BatchNorm Layers in the direction vectors")
    parser.add_argument("--plot_before", action="store_true", default=False,
                        help="Draw a surface of the original network.")
    parser.add_argument("--use_checkpoint", action="store_true", default=False,
                        help="Specify to use a checkpoint. If there is one, all Experiment Configurations are ignored")

    # Model Configuration
    parser.add_argument("--model", "-m", type=str, default="resnet18COB", choices=get_model_names())
    parser.add_argument("--save_model_path", type=str, default=None,
                        help="save path for the selected network")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="pt file location for the selected network")

    return parser.parse_args()


def generate_new_direction_vectors(net, use_bias_bn):
    if not use_bias_bn:
        weights = net.get_weights(flatten=False, concat=False)
        delta = generate_random_2d_vector(weights, ignore_bias_bn=True, seed=123)
        eta = generate_random_2d_vector(weights, ignore_bias_bn=True, seed=321)

        weights = torch.cat([w.flatten() for w in weights])
        delta = torch.cat([d.flatten() for d in delta])
        eta = torch.cat([d.flatten() for d in eta])
    else:
        weights = net.get_weights()
        delta = generate_random_2d_vector(weights, seed=123)
        eta = generate_random_2d_vector(weights, seed=321)

    return (delta, eta), weights


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    args = argument_parser()

    checkpoint_exist = pathlib.Path(checkpoint_file).exists() and args.use_checkpoint

    x_coordinates = torch.linspace(args.x[0], args.x[1], int(args.x[2]))
    y_coordinates = torch.linspace(args.y[0], args.y[1], int(args.y[2]))
    surface = [(x, y) for x in x_coordinates for y in y_coordinates]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset, valset, testset = get_dataset_subsets(args.dataset)
    metric = TrainingMetrics(
        nn.CrossEntropyLoss(),
        [accuracy])
    config = TrainingConfig(
        optimizer=(args.optimizer, {"lr": args.lr}),
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size)

    net = get_model(args.dataset, args.model, device=device)
    if args.load_model_path:
        load_dict = torch.load(args.load_model_path)
        if not net.state_dict().keys() == load_dict.keys():
            raise Exception("Model that was loaded does not match the model type used in the experiment.")
        net.load_state_dict(load_dict)

    else:
        if args.train:
            train(net, train_dataset=trainset, metrics=metric, config=config)
            test(net, dataset=trainset, metrics=metric, config=config)
        if args.save_model:
            torch.save(net.state_dict(), get_nonexistent_path(args.save_path))

    checkpoint = None
    if checkpoint_exist:
        print("A checkpoint exists and is requested to use, overriding all Experiment configuration!")
        checkpoint = torch.load(checkpoint_file)
        step = checkpoint['step']
        surface = checkpoint['surface'][step:]
        section = checkpoint['section']

    plot_before = args.plot_before if not checkpoint else section == "before"
    if plot_before:
        direction, original_weights = generate_new_direction_vectors(net, args.use_bias_bn)
        try:
            loss_surf_before, _ = generate_contour_loss_values(net, direction, original_weights, surface,
                                                               trainset, metric, config, checkpoint)
            torch.save({"before_loss_surface": loss_surf_before},
                       "/tmp/{}_{}_before_loss_surface.pth".format(args.model, args.cob_range))
            plot_contours(x_coordinates, y_coordinates, loss_surf_before)
        except Exception:
            checkpoint = torch.load(checkpoint_file)
            checkpoint['section'] = "before"
            torch.save(checkpoint, checkpoint_file)
            exit()

        # This is to force the model to go back to the original state
        # since the contour generator modified the weights of the net.
        net.set_weights(original_weights)

    net = net.random_teleport(cob_range=args.cob_range, sampling_type=args.cob_sampling)
    direction, teleported_weights = generate_new_direction_vectors(net, args.use_bias_bn)
    if checkpoint and plot_before:
        # If a checkpoint has been loaded, the checkpoint redefine the surface length.
        # Thus we need to get the original surface.
        surface = [(x, y) for x in x_coordinates for y in y_coordinates]
    try:
        loss_surf_after, acc = generate_contour_loss_values(net, direction, teleported_weights, surface,
                                                            trainset, metric, config, checkpoint)
        torch.save({"after_loss_surface": loss_surf_after},
                   "/tmp/{}_{}_after_loss_surface.pth".format(args.model, args.cob_range))
        plot_contours(x_coordinates, y_coordinates, loss_surf_after)
    except Exception:
        checkpoint = torch.load(checkpoint_file)
        checkpoint['section'] = "after"
        torch.save(checkpoint, checkpoint_file)
        exit()
    plt.show()

    if checkpoint_exist:
        pathlib.Path(checkpoint_file).unlink()
