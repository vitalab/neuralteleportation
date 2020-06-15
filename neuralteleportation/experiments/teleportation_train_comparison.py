import argparse
from dataclasses import dataclass

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
from neuralteleportation.training.experiment_setup import get_model_list, get_model_from_string
from neuralteleportation.training.experiment_setup import get_cifar10_datasets, get_mnist_datasets
from neuralteleportation.training.training import train_epoch, test

__models__ = get_model_list()


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
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--teleport_chance", tpye=float, default=0.5,
                        help="probability of teleporting in the 2nd scenario")
    parser.add_argument("--teleport_every", type=int, default=4,
                        help="After how many epoch should the model be teleported")
    parser.add_argument("--cob_range", type=float, default=0.5, help="Sets the range of change of basis")
    parser.add_argument("--cob_sampling", type=str, default="usual",
                        choices=["usual", "symmetric", "negative", "zero"],
                        help="What type of sampling should be used for the teleportation")
    parser.add_argument("--plot", action="store_true", default=False, help="")
    parser.add_argument("--targeted_teleportation", action="store_true", default=False,
                        help="Specify if the teleportation should use a specific change of basis or use a random one.")

    return parser.parse_args()


def generate_experience_models(model: str, device: str = 'cpu') -> tuple:
    """
        Creates 3 models base on the same initial set of weights.
    """
    net_vanilla = get_model_from_string(model, num_classes=10, input_channels=dims).to(device)
    net_5050 = copy.deepcopy(net_vanilla)
    net_5050 = NeuralTeleportationModel(network=net_5050,
                                        input_shape=(args.batch_size, dims, w, h)).to(device).to(device)
    net_teleport = copy.deepcopy(net_vanilla)
    net_teleport = NeuralTeleportationModel(network=net_teleport,
                                            input_shape=(args.batch_size, dims, w, h)).to(device).to(device)

    return net_vanilla, net_5050, net_teleport


def start_training(model: NeuralTeleportationModel,
                   trainloader: DataLoader,
                   valset: VisionDataset,
                   metric: TrainingMetrics,
                   config: CompareTrainingConfig,
                   teleport_chance: float) -> np.ndarray:
    """
        This function start a model training with a specific scerio configuraions.

        Scenario 1: train the model without using teleportation (teleportation_chance = 0.0)
        Scenario 2: train the model using a probability of teleporting teleportation every Xth epochs
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

    net1, net2, net3 = generate_experience_models(args.model, device=device)

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

    res_vanilla = []
    res_5050 = []
    res_teleport = []

    val_res = test(model=net1, dataset=valset, metrics=metric, config=config)
    res_vanilla.append(val_res['accuracy'])
    res_5050.append(val_res['accuracy'])
    res_teleport.append(val_res['accuracy'])

    print()
    print("===============================")
    print("======== Training =============")
    print("===============================")
    for _ in range(args.run):
        res_vanilla = np.concatenate((res_vanilla,
                                      start_training(net1, trainloader, valset, metric, config, teleport_chance=0.0)))

    print("===============================")
    print("======== Training 5050=========")
    print("===============================")
    for _ in range(args.run):
        res_5050 = np.concatenate((res_5050,
                                   start_training(net2, trainloader, valset, metric, config, teleport_chance=0.5)))

    print("===============================")
    print("======== Training Teleport=====")
    print("===============================")
    for _ in range(args.run):
        res_teleport = np.concatenate((res_teleport,
                                       start_training(net3, trainloader, valset, metric, config, teleport_chance=1.0)))

    title = "SGD %s epochs training at %e, %s, %s" % (args.epochs, args.lr, args.model, args.dataset)
    if args.plot:
        plt.figure()
        plt.title(title)
        plt.plot(res_vanilla, '-o', label="vanilla", markersize=3)
        plt.plot(res_5050, '-o', label="net_5050", markersize=3)
        plt.plot(res_teleport, '-o', label="net_teleport", markersize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xlim([0, args.epochs])
        plt.ylim([0, 1])
        for x in np.arange(args.teleport_every, args.epochs, args.teleport_every):
            plt.axvline(x, linestyle='--', color='b')
        plt.legend()
        plt.show()

    root = pathlib.Path().absolute()
    file_path = root / ("results/" + title + ".h5").replace(" ", "_").replace(",", "")

    f = h5py.File(file_path, 'a')
    f.create_dataset("vanilla", data=res_vanilla)
    f.create_dataset("5050", data=res_5050)
    f.create_dataset("teleport", data=res_teleport)

    f.close()
