import argparse
import numpy as np

import torch
import torch.nn as nn

from neuralteleportation.metrics import accuracy
from neuralteleportation.losslandscape import losslandscape
from neuralteleportation.losslandscape.losslandscape import LandscapeConfig
from neuralteleportation.training.training import TrainingMetrics
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.experiment_setup import get_cifar10_datasets, resnet18COB


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the training")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="for how many epochs should the model train")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="used batch size.")
    parser.add_argument("--cob_range", type=float, default=0.5,
                        help="Defines the range used for the COB. It must be a valid mix with cob_sampling")
    parser.add_argument("--cob_sampling", type=str, default="usual",
                        help="Defines the type of sampling used for the COB. It must be a valide mix with cob_range")
    parser.add_argument("--teleport_at", "-t", type=int, default=5,
                        help="Make the model teleport after at the given epoch number")
    parser.add_argument("--x", type=int, default="20",
                        help="Defines the precision of the x-axis")
    parser.add_argument("--y", type=int, default="20",
                        help="Defines the precision of the y-axis")
    parser.add_argument("--scope", type=float, default=1.0,
                        help="Apply the factor to the direction vector in order to get specific scopes.")
    parser.add_argument("--model", type=str, default="resnet18COB",
                        help="Defines what model to plot the surface of.")
    parser.add_argument("--use_teleport_direction", action="store_true", default=False,
                        help="If the direction vector from the teleportation should be used as X-axis direction vector")

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metric = TrainingMetrics(
        criterion=nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = LandscapeConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cob_range=args.cob_range,
        cob_sampling=args.cob_sampling,
        teleport_at=args.teleport_at,
        device=device
    )

    model = resnet18COB(num_classes=10)
    trainset, valset, testset = get_cifar10_datasets()
    # trainset.data = trainset.data[:5000]

    x = torch.linspace(-1, 1, args.x)
    y = torch.linspace(-1, 1, args.y)
    shape = x.shape if y is None else (len(x), len(y))
    surface = torch.stack((x, y))

    model = NeuralTeleportationModel(model, input_shape=(config.batch_size, 3, 32, 32)).to(device)
    original_w = model.get_weights()
    w_checkpoints, final_w = losslandscape.generate_teleportation_training_weights(model, trainset,
                                                                                   metric=metric, config=config)
    delta = losslandscape.generate_random_2d_vector(final_w, seed=1) * args.scope
    if args.use_teleport_direction:
        delta = losslandscape.generate_direction_vector(w_checkpoints, args.teleport_at)
    eta = losslandscape.generate_random_2d_vector(final_w, seed=2) * args.scope

    # Calculate angle between the two direction vectors.
    print("angle between direction is {} rad".format(losslandscape.compute_angle(delta, eta)))

    loss, _ = losslandscape.generate_contour_loss_values(model, (delta, eta), surface, trainset, metric, config)
    loss = np.array(loss)
    loss = np.resize(loss, shape)

    w_diff = [(w - final_w) for w in w_checkpoints]
    w_x_dirrection, w_y_dirrection = losslandscape.generate_weights_direction(original_w, w_diff)
    weight_traj = losslandscape.generate_weight_trajectory(w_diff, (w_x_dirrection, w_y_dirrection))

    teleport_idx = np.arange(config.teleport_every + 1, len(w_checkpoints), config.teleport_every + 1, dtype=np.int)
    losslandscape.plot_contours(x, y, loss, weight_traj, teleport_idx)
