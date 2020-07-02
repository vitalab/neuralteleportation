import argparse

import torch
import torch.nn as nn
from torch.cuda import is_available as cuda_avail

from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
from neuralteleportation.training.training import train
from neuralteleportation.training.experiment_setup import get_cifar10_datasets
from neuralteleportation.training.config import TrainingMetrics
from neuralteleportation.losslandscape import losslandscape as ll
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.metrics import accuracy


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=3,
                        help="How many epochs should the network train in total")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Defines how big the batch size is")
    parser.add_argument("--cob_range", type=float, default=0.5,
                        help="Defines the range used for the COB. It must be a valid mix with cob_sampling")
    parser.add_argument("--cob_sampling", type=str, default="usual",
                        help="Defines the type of sampling used for the COB. It must be a valide mix with cob_range")
    parser.add_argument("--x", nargs=3, type=int, default=[0, 1, 50],
                        help="Defines the precision of the alpha")
    parser.add_argument("--train_model", action="store_true", default=False,
                        help="Whether or not the model should train before teleportation.")

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    device = 'cuda' if cuda_avail() else 'cpu'

    trainset, valset, testset = get_cifar10_datasets()
    # Uncommented for debug
    # trainset.data = trainset.data[:100]

    model = NeuralTeleportationModel(resnet18COB(num_classes=10), input_shape=(args.batch_size, 3, 32, 32)).to(device)
    metric = TrainingMetrics(
        criterion=nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = ll.LandscapeConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        cob_range=args.cob_range,
        cob_sampling=args.cob_sampling,
        teleport_at=[5],
        device=device
    )
    if args.train_model:
        train(model, trainset, metric, config)
    a = torch.linspace(args.x[0], args.x[1], args.x[2])
    w_o = model.get_weights()
    model.random_teleport(args.cob_range, args.cob_sampling)
    w_t = model.get_weights()
    loss, acc = ll.generate_1D_linear_interp(model, w_o, w_t, a, metric=metric, config=config, trainset=trainset)
    ll.plot_interp(loss, acc, a)
