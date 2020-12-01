import argparse
import numpy as np

import torch
import torch.nn as nn

from neuralteleportation.metrics import accuracy
from neuralteleportation.losslandscape import losslandscape
from neuralteleportation.losslandscape.losslandscape import LandscapeConfig
from neuralteleportation.training.training import TrainingMetrics
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.experiment_setup import get_dataset_subsets, get_model, get_model_names


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the training")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="Select the Gradient Descente Optimizer for the model's training.")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="for how many epochs should the model train")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="used batch size.")
    parser.add_argument("--cob_range", type=float, default=0.5,
                        help="Defines the range used for the COB. It must be a valid mix with cob_sampling")
    parser.add_argument("--cob_sampling", type=str, default="intra_landscape",
                        help="Defines the type of sampling used for the COB. It must be a valide mix with cob_range")
    parser.add_argument("--teleport_at", "-t", nargs='+', type=int, default=[5],
                        help="Make the model teleport after at the given epoch number")
    parser.add_argument("--x", type=int, default=50,
                        help="Defines the precision of the x-axis")
    parser.add_argument("--y", type=int, default=50,
                        help="Defines the precision of the y-axis")
    parser.add_argument("--model", type=str, default="resnet18COB", choices=get_model_names(),
                        help="Defines what model to plot the surface of.")
    parser.add_argument("--show_weight_traj", action="store_true", default=False,
                        help="Enable the plotting of the weights trajectory while training.")

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metric = TrainingMetrics(
        criterion=nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = LandscapeConfig(
        optimizer=(args.optimizer,{"lr": args.lr}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        cob_range=args.cob_range,
        cob_sampling=args.cob_sampling,
        teleport_at=args.teleport_at,
        device=device
    )

    trainset, valset, testset = get_dataset_subsets("cifar10")
    model = get_model("cifar10", args.model)

    x = torch.linspace(-1, 1, args.x)
    y = torch.linspace(-1, 1, args.y)
    shape = x.shape if y is None else (len(x), len(y))
    surface = torch.stack((x, y))

    model.to(device)
    model = NeuralTeleportationModel(model, input_shape=(config.batch_size, 3, 32, 32)).to(device)
    original_w = model.get_weights()
    w_checkpoints, final_w = losslandscape.generate_teleportation_training_weights(model, trainset,
                                                                                   metric=metric, config=config)
    delta = losslandscape.generate_random_2d_vector(final_w, seed=1)
    eta = losslandscape.generate_random_2d_vector(final_w, seed=2)

    # Calculate angle between the two direction vectors.
    print("angle between direction is {} rad".format(losslandscape.compute_angle(delta, eta)))

    loss, _ = losslandscape.generate_contour_loss_values(model, (delta, eta), surface, trainset, metric, config)
    loss = np.array(loss)
    loss = np.resize(loss, shape)

    w_diff = [(w - final_w) for w in w_checkpoints]
    weight_traj = None
    if args.show_weight_traj:
        w_x_dirrection, w_y_dirrection = losslandscape.generate_weights_direction(original_w, w_diff)
        weight_traj = losslandscape.generate_weight_trajectory(w_diff, (w_x_dirrection, w_y_dirrection))

    teleport_idx = [i + (n + 1) for n, i in enumerate(args.teleport_at)]
    losslandscape.plot_contours(x, y, loss, weight_traj, teleport_idx)
