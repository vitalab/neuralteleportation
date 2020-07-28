import torch
import torch.nn as nn
import argparse

from neuralteleportation.training.experiment_setup import get_dataset_subsets, get_model, get_model_names
from neuralteleportation.training.training import train, test
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.metrics import accuracy
from neuralteleportation.losslandscape import generate_random_2d_vector, generate_contour_loss_values, plot_contours


def argument_parser():
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the plot loss landscape experiment.')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10,
                        help='How many epoch to train the network if train set to true')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate used for the training if train is active.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="Select the Gradient descent optimizer for the experiment")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"],
                        help="Selects the dataset use in the experiment")
    parser.add_argument("--model", "-m", type=str, default="resnet18COB", choices=get_model_names())
    parser.add_argument("--train", "-t", action="store_true", default=False,
                        help="if the model should be train before teleportation")
    parser.add_argument("--load_model", type=str, default="", help="file path of the h5 network state file.")
    parser.add_argument("--save_model", "-s", action="store_true", default=False, help="tell to save the model or not")
    parser.add_argument("--save_model_location", type=str, default='/tmp/model.pt',
                        help="save path and .pt file forthe selected network")
    parser.add_argument("--cob_range", type=float, default=0.5, help='set the CoB range for the teleportation.')
    parser.add_argument("--cob_sampling", type=str, default="within_landscape", help="Sampling type for CoB.")
    parser.add_argument("--x", nargs=3, type=float, default=[-1, 1, 31],
                        help="Defines the precision of the x")
    parser.add_argument("--y", nargs=3, type=float, default=[-1, 1, 31],
                        help="Defines the precision of the y")
    parser.add_argument("--use_biasbn", action="store_true", default=False,
                        help="Wether or not to consider bias in layer and BatchNorm Layers in the direction vectors")
    parser.add_argument("--plot_before", action="store_true", default=False,
                        help="Draw a surface of the original network.")

    return parser.parse_args()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    args = argument_parser()

    x = torch.linspace(args.x[0], args.x[1], int(args.x[2]))
    y = torch.linspace(args.y[0], args.y[1], int(args.y[2]))
    surface = torch.stack((x, y))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset, valset, testset = get_dataset_subsets(args.dataset)
    net = get_model(args.dataset, args.model, device=device)

    metric = TrainingMetrics(
        nn.CrossEntropyLoss(),
        [accuracy])
    config = TrainingConfig(
        optimizer=(args.optimizer, {"lr": args.lr}),
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size)

    if args.train:
        train(net, train_dataset=trainset, metrics=metric, config=config)
        test(net, dataset=trainset, metrics=metric, config=config)

    if args.save_model:
        torch.save(net, args.save_model_location)

    if args.plot_before:
        if not args.use_biasbn:
            w_o = net.get_weights(flatten=False, concat=False)
            delta = generate_random_2d_vector(w_o, ignore_biasbn=True, seed=123)
            eta = generate_random_2d_vector(w_o, ignore_biasbn=True, seed=321)

            w_o = torch.cat([w.flatten() for w in w_o])
            delta = torch.cat([d.flatten() for d in delta])
            eta = torch.cat([d.flatten() for d in eta])
        else:
            w_o = net.get_weights()
            delta = generate_random_2d_vector(w_o, seed=123)
            eta = generate_random_2d_vector(w_o, seed=321)
        direction = delta, eta
        loss_surf_before, _ = generate_contour_loss_values(net, direction, w_o,
                                                           surface, trainset, metric, config)
        plot_contours(x, y, loss_surf_before)
        # this is to force the model to get to the original state
        # since the contour generator modified the weights of the net.
        net.set_weights(w_o)

    net = net.random_teleport(cob_range=args.cob_range, sampling_type=args.cob_sampling)
    if not args.use_biasbn:
        w_t = net.get_weights(flatten=False, concat=False)
        delta = generate_random_2d_vector(w_t, ignore_biasbn=True, seed=123)
        eta = generate_random_2d_vector(w_t, ignore_biasbn=True, seed=321)

        w_t = torch.cat([w.flatten() for w in w_t])
        delta = torch.cat([d.flatten() for d in delta])
        eta = torch.cat([d.flatten() for d in eta])
    else:
        w_t = net.get_weights()
        delta = generate_random_2d_vector(w_t, seed=123)
        eta = generate_random_2d_vector(w_t, seed=321)
    direction = delta, eta
    loss_surf_after, acc = generate_contour_loss_values(net, direction, w_t, surface, trainset, metric, config)
    plot_contours(x, y, loss_surf_after)

    plt.show()
