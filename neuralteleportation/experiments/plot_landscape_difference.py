import argparse
from copy import deepcopy

import torch
import matplotlib.pyplot as plt

from neuralteleportation.losslandscape import generate_1D_linear_interp, plot_interp
from neuralteleportation.training.experiment_setup import get_model_names, get_model, get_dataset_subsets
from neuralteleportation.training.training import TrainingMetrics, TrainingConfig
from neuralteleportation.metrics import accuracy


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", nargs=3, type=float, default=[-0.5, 1.5, 401],
                        help="Defines the precision of the alpha")
    parser.add_argument("--cob_range", type=float, default=0.5,
                        help="Defines the range used for the COB. It must be a valid mix with cob_sampling")
    parser.add_argument("--model", type=str, default="resnet18COB", choices=get_model_names())
    parser.add_argument("--load_model_path", type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metric = TrainingMetrics(
        criterion=torch.nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )
    config = TrainingConfig(
        device=device
    )

    trainset, valset, testset = get_dataset_subsets("cifar10")

    # W represent the original network.
    w = get_model("cifar10", args.model, device=device)

    if args.load_model_path:
        state = torch.load(args.load_model_path)
        w.load_state_dict(state)

    # Teleport model within landscape first with cob_range and save the model for later use.
    # Then teleport again the original network,
    tw = w.random_teleport(cob_range=args.cob_range, sampling_type="within_landscape", inline=False)
    w_prime = w.random_teleport(cob_range=args.cob_range, sampling_type="change_landscape", inline=False)

    # Once both teleported model are saved,
    # apply the cob of the change landscape teleported network
    # to the within landscape teleported model.
    cob_cl = w_prime.get_cob()
    tw_prime = tw.teleport(cob=cob_cl, inline=False, reset_teleportation=False)

    alpha = torch.linspace(args.x[0], args.x[1], int(args.x[2]))

    # Plot for the model and teleport within landscape
    loss, acc_t, acc_v = generate_1D_linear_interp(w, w.get_params(), tw.get_params(),
                                                   alpha, trainset=trainset, valset=valset,
                                                   metric=metric, config=config)
    plot_interp(loss, acc_train=acc_t, a=alpha, acc_val=acc_v)

    torch.save({
        'loss': loss,
        'acc_t': acc_t,
        'acc_v': acc_v,
        'alpha': alpha
    }, "/tmp/{}_cob_range_{}_WL_linterp.pth".format(args.model, args.cob_range))

    # Plot for the model teleported with change landscape and
    # the within landscape teleported model teleported to the change landscape sampling.
    loss, acc_t, acc_v = generate_1D_linear_interp(w_prime, w_prime.get_params(), tw_prime.get_params(),
                                                   alpha, trainset=trainset, valset=valset,
                                                   metric=metric, config=config)
    plot_interp(loss, acc_train=acc_t, a=alpha, acc_val=acc_v)
    torch.save({
        'loss': loss,
        'acc_t': acc_t,
        'acc_v': acc_v,
        'alpha': alpha
    }, "/tmp/{}_cob_range_{}_CL_linterp.pth".format(args.model, args.cob_range))
    plt.show()
