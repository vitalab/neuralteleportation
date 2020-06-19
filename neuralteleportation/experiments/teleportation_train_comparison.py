import argparse
from dataclasses import dataclass
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import pathlib
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from neuralteleportation.metrics import accuracy
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.config import TrainingConfig, TrainingMetrics
from neuralteleportation.training.experiment_setup import get_model_names, get_model_from_name
from neuralteleportation.training.experiment_setup import get_cifar10_datasets, get_mnist_datasets
from neuralteleportation.training.training import train_epoch, test

__models__ = get_model_names()


@dataclass
class CompareTrainingConfig(TrainingConfig):
    targeted_teleportation: bool = False
    teleport_every_n_epochs: int = 4
    cob_range: float = 0.5
    cob_sampling: str = "usual"


def argumentparser():
    parser = argparse.ArgumentParser(description="Simple argument parser for traininng teleportation")
    parser.add_argument("--epochs", type=int, default=50, help="How many epochs should all models train on")
    parser.add_argument("--run", type=int, default=1, help="How many times should a scenario be run.")
    parser.add_argument("--model", type=str, default="resnet18COB", choices=__models__,
                        help="Which model should be train")
    parser.add_argument("--lr", type=float, default=1e-3, help="The applied learning rate for all models when training")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--teleport_chance", type=float, default=0.5,
                        help="probability of teleporting in the 2nd scenario")
    parser.add_argument("--teleport_every", type=int, default=4,
                        help="After how many epoch should the model be teleported")
    parser.add_argument("--cob_range", type=float, default=0.5, help="Sets the range of change of basis")
    parser.add_argument("--cob_sampling", type=str, default="within_landscape",
                        choices=["within_landscape", "change_landscape", "positive", "negative", "centered"],
                        help="What type of sampling should be used for the teleportation")
    parser.add_argument("--plot", action="store_true", default=False, help="")
    parser.add_argument("--targeted_teleportation", action="store_true", default=False,
                        help="Specify if the teleportation should use a specific change of basis or use a random one.")

    return parser.parse_args()


def generate_experience_models(model: str, batch_size, input_dims: Tuple[int, int, int], device: str = 'cpu') -> tuple:
    """
        Creates 3 models based on the same initial set of weights.
    """
    w, h, dims = input_dims
    net1 = get_model_from_name(model, num_classes=10, input_channels=dims)
    net1 = NeuralTeleportationModel(network=net1,
                                    input_shape=(batch_size, dims, w, h)).to(device)

    net2 = copy.deepcopy(net1)
    net2 = NeuralTeleportationModel(network=net2,
                                    input_shape=(batch_size, dims, w, h)).to(device)
    net3 = copy.deepcopy(net1)
    net3 = NeuralTeleportationModel(network=net3,
                                    input_shape=(args.batch_size, dims, w, h)).to(device)

    return net1, net2, net3


def start_training(model: NeuralTeleportationModel,
                   trainloader: DataLoader,
                   valset: VisionDataset,
                   metric: TrainingMetrics,
                   config: CompareTrainingConfig,
                   teleport_chance: float) -> np.ndarray:
    """
        This function starts a model training with a specific Scenario configuration.

        Scenario 1: train the model without using teleportation (teleportation_chance = 0.0)
        Scenario 2: train the model using a probability of teleporting every Xth epochs
        (0 < teleportation_chance < 1.0)
        Scenario 3: train the model using teleportation every Xth epochs (teleportation_chance = 1.0)

        returns:
            np.array containing the validation accuracy results of every epochs.
    """
    model.to(config.device)
    optimizer = torch.optim.SGD(lr=config.lr, params=model.parameters())

    results = []
    for e in np.arange(1, args.epochs + 1):
        train_epoch(model=model,
                    criterion=metric.criterion,
                    optimizer=optimizer,
                    train_loader=trainloader,
                    epoch=e,
                    device=config.device)
        results.append(test(model=model, dataset=valset, metrics=metric, config=config)['accuracy'])
        model.train()

        if e % config.teleport_every_n_epochs == 0 and random.random() <= teleport_chance:
            print("teleported model")
            if config.targeted_teleportation:
                # TODO: use teleportation function here when they are available.
                raise NotImplementedError
            else:
                model.random_teleport(cob_range=config.cob_range, sampling_type=config.cob_sampling)
                optimizer = torch.optim.SGD(lr=config.lr, params=model.parameters())

    model.cpu()  # Force the network to go out of the cuda mem.

    return np.array(results)


if __name__ == '__main__':
    args = argumentparser()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    if args.dataset == 'cifar10':
        trainset, valset, testset = get_cifar10_datasets()
    elif args.dataset == 'mnist':
        trainset, valset, testset = get_mnist_datasets()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)

    w, h = trainset.data.shape[1:3]
    dims = 1 if trainset.data.ndim < 4 else trainset.data.shape[3]

    nets = generate_experience_models(args.model, args.batch_size, (w, h, dims), device=device)
    init_weights = nets[0].get_weights()

    metric = TrainingMetrics(criterion=nn.CrossEntropyLoss(), metrics=[accuracy])
    config = CompareTrainingConfig(lr=args.lr,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   device=device,
                                   cob_range=args.cob_range,
                                   cob_sampling=args.cob_sampling,
                                   targeted_teleportation=args.targeted_teleportation,
                                   teleport_every_n_epochs=args.teleport_every
                                   )

    res_vanilla = np.empty((args.run, args.epochs + 1))
    res_5050 = np.empty((args.run, args.epochs + 1))
    res_teleport = np.empty((args.run, args.epochs + 1))

    results = [res_vanilla, res_5050, res_teleport]

    # No need to run test for each since they all have the same weights at start.
    init_val_res = test(model=nets[0], dataset=valset, metrics=metric, config=config)['accuracy']
    teleport_probs = [0.0, args.teleport_chance, 1, 0]
    for net in nets:
        scenarion_num = nets.index(net)
        print("Starting scenario {}".format(scenarion_num + 1))
        for n in range(args.run):
            print("run no {}".format(n + 1))
            net.set_weights(init_weights)
            results[scenarion_num][n] = np.concatenate(([init_val_res],
                                                        start_training(net, trainloader, valset, metric, config,
                                                                       teleport_chance=teleport_probs[scenarion_num])))

    mean_vanilla = res_vanilla.mean(axis=0)
    std_vanilla = res_vanilla.std(axis=0) if args.run > 1 else 0

    mean_5050 = res_5050.mean(axis=0)
    std_5050 = res_5050.std(axis=0) if args.run > 1 else np.zeros(args.epochs+1)

    mean_teleport = res_teleport.mean(axis=0)
    std_teleport = res_teleport.std(axis=0) if args.run > 1 else np.zeros(args.epochs+1)

    title = "SGD %s epochs training at %.1e, %s, %s" % (args.epochs, args.lr, args.model, args.dataset)
    if args.plot:
        x = np.arange(args.epochs + 1)
        plt.figure()
        plt.title(title)
        plt.errorbar(x, mean_vanilla, yerr=std_vanilla, fmt='-o', label="vanilla", markersize=3)
        plt.errorbar(x, mean_5050, yerr=std_vanilla, fmt='-o', label="net_5050", markersize=3)
        plt.errorbar(x, mean_teleport, yerr=std_vanilla, fmt='-o', label="net_teleport", markersize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xlim([0, args.epochs])
        for x in np.arange(args.teleport_every, args.epochs, args.teleport_every):
            plt.axvline(x, linestyle='--', color='b')
        plt.legend()
        plt.show()

    root = pathlib.Path().absolute()
    file_path = root / "results/"

    if not file_path.exists():
        file_path.mkdir()

    file_path = file_path.joinpath((title + ".h5").replace(" ", "_").replace(",", ""))
    f = h5py.File(file_path, 'a')

    # Flush the existing data in the existing file.
    if file_path.exists():
        for k in f.keys():
            del f[k]

    f.create_dataset("vanilla", data=res_vanilla)
    f.create_dataset("5050", data=res_5050)
    f.create_dataset("teleport", data=res_teleport)

    f.close()
