import argparse

import torch
import torch.nn as nn
from torch.cuda import is_available as cuda_avail

from neuralteleportation.training.training import train
from neuralteleportation.training.experiment_setup import get_dataset_subsets, get_model, get_model_names
from neuralteleportation.training.config import TrainingMetrics
from neuralteleportation.losslandscape.losslandscape import LandscapeConfig, generate_1D_linear_interp, plot_interp
from neuralteleportation.metrics import accuracy


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="How many epochs should the network train in total")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="The learning rate for model training")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="Select the used Optimizer during model training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Defines how big the batch size is")
    parser.add_argument("--cob_range", type=float, default=0.5,
                        help="Defines the range used for the COB. It must be a valid mix with cob_sampling")
    parser.add_argument("--cob_sampling", type=str, default="intra_landscape",
                        choices=['intra_landscape', 'inter_landscape', 'positive', 'negative', 'centered'],
                        help="Defines the type of sampling used for the COB. It must be a valide mix with cob_range")
    parser.add_argument("--x", nargs=3, type=float, default=[-0.5, 1.5, 101],
                        help="Defines the precision of the alpha")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Whether or not the model should train before teleportation.")
    parser.add_argument("--model", type=str, default="resnet18COB", choices=get_model_names())

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = 'cuda' if cuda_avail() else 'cpu'

    trainset, valset, testset = get_dataset_subsets("cifar10")

    model = get_model("cifar10", args.model, device=device)
    metric = TrainingMetrics(
        criterion=nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = LandscapeConfig(
        optimizer=(args.optimizer, {"lr": args.lr}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        cob_range=args.cob_range,
        cob_sampling=args.cob_sampling,
        teleport_at=[args.epochs],
        device=device
    )
    if args.train:
        train(model, trainset, metric, config)
    a = torch.linspace(args.x[0], args.x[1], int(args.x[2]))
    param_o = model.get_params()
    model.random_teleport(args.cob_range, args.cob_sampling)
    param_t = model.get_params()

    loss, acc_t, loss_v, acc_v = generate_1D_linear_interp(model, param_o, param_t, a,
                                                   metric=metric, config=config,
                                                   trainset=trainset, valset=valset)
    plot_interp(loss, acc_t, a, "W", "T(W)", acc_val=acc_v, title="Linear Interpolation between W and T(W)")
