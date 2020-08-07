import argparse
import pathlib

import torch
import torch.nn as nn
from torch.cuda import is_available as cuda_avail

from neuralteleportation.metrics import accuracy
from neuralteleportation.training.training import train, test
from neuralteleportation.training.experiment_setup import get_dataset_subsets, get_model, get_model_names
from neuralteleportation.training.config import TrainingMetrics
from neuralteleportation.losslandscape.losslandscape import LandscapeConfig, generate_1D_linear_interp, plot_interp

from neuralteleportation.losslandscape import linterp_checkpoint_file as checkpoint_file


def argument_parser():
    parser = argparse.ArgumentParser()

    """Hyper Parameters"""
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
    parser.add_argument("--cob_sampling", type=str, default="within_landscape",
                        choices=['within_landscape', 'change_landscape', 'positive', 'negative', 'centered'],
                        help="Defines the type of sampling used for the COB. It must be a valide mix with cob_range")

    """Experiment Config"""
    parser.add_argument("--x", nargs=3, type=float, default=[-0.5, 1.5, 101],
                        help="Defines the precision of the alpha")
    parser.add_argument("--use_checkpoint", action="store_true", default=False,
                        help="Specify to use a checkpoint if there is one")

    """Models Configuration"""
    parser.add_argument("--train", action="store_true", default=False,
                        help="Whether or not the model should train before teleportation.")
    parser.add_argument("--model", type=str, default="resnet18COB", choices=get_model_names())
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="Enable saving the model after training")
    parser.add_argument("--save_path", type=str, default='/tmp/model.pt',
                        help="")
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default='/tmp/model.pt')

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    checkpoint_exist = pathlib.Path(checkpoint_file).exists() and args.use_checkpoint

    device = 'cuda' if cuda_avail() else 'cpu'

    trainset, valset, testset = get_dataset_subsets("cifar10")
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

    model = get_model("cifar10", args.model, device=device)
    if args.load_model:
        load_dict = torch.load(args.load_path)
        if not model.state_dict().keys() == load_dict.keys():
            raise Exception("Model that was loaded does not match the model type used in the experiment.")
        model.load_state_dict(load_dict)
        # res = test(model, trainset, metric, config)
        # print("Scored {:.4f} acc on trainset".format(res['accuracy']))
        # res = test(model, valset, metric, config)
        # print("Scored {:.4f} acc on valset".format(res['accuracy']))
    else:
        if args.train:
            train(model, trainset, metric, config)
            test(model, testset, metric, config)
        if args.save_model:
            torch.save(model.state_dict(), args.save_path)

    a = torch.linspace(args.x[0], args.x[1], int(args.x[2]))
    param_o, param_t = None, None
    checkpoint = None
    if checkpoint_exist:
        print("A checkpoint existe and is requested to use, overriding all Experiment configuration!")
        checkpoint = torch.load(checkpoint_file)
        step = checkpoint['step']
        a = checkpoint['alpha'][step:]
        param_o = checkpoint['original_model']
        param_t = checkpoint['teleported_model']
    else:
        param_o = model.get_params()
        model.random_teleport(args.cob_range, args.cob_sampling)
        param_t = model.get_params()

    loss, acc_t, acc_v = generate_1D_linear_interp(model, param_o, param_t, a, metric=metric, config=config,
                                                   trainset=trainset, valset=valset)

    torch.save(checkpoint, '/tmp/linterp_save_checkpoint.pth')
    if checkpoint:
        loss = checkpoint['losses'].append(loss)
        acc_t = checkpoint['acc_t'].append(acc_t)
        acc_v = checkpoint['acc_v'].append(acc_v)
    plot_interp(loss, acc_t, a, acc_val=acc_v)

    if checkpoint_exist:
        pathlib.Path(checkpoint_file).unlink()
