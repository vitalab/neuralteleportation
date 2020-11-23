import argparse

import torch
import torch.nn as nn
from torch.cuda import is_available as cuda_avail
import json

from os.path import join as pjoin

from neuralteleportation.training.training import train, test
from neuralteleportation.training.experiment_setup import get_dataset_subsets, get_model, get_model_names
from neuralteleportation.training.config import TrainingMetrics
from neuralteleportation.losslandscape.losslandscape import LandscapeConfig, generate_1D_linear_interp, plot_interp
from neuralteleportation.metrics import accuracy
from neuralteleportation.utils.pathutils import get_nonexistent_path


def argument_parser():
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument("--train", action="store_true", default=True,
                        help="Whether or not the models should train before interpolation.")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="How many epochs should the networks train in total")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate for model training")
    parser.add_argument("--optimizerA", type=str, default="SGD", choices=["Adam", "SGD"],
                        help="Optimizer used by model A")
    parser.add_argument("--optimizerB", type=str, default="SGD", choices=["Adam", "SGD"],
                        help="Optimizer used by model B")
    parser.add_argument("--batch_sizeA", type=int, default=8, help="Defines how big the batch size is")
    parser.add_argument("--batch_sizeB", type=int, default=1024, help="Defines how big the batch size is")

    # Teleportation params
    parser.add_argument("--teleportA", action="store_true", default=True,
                        help="Whether or not to teleport model A for second interpolation")
    parser.add_argument("--teleportB", action="store_true", default=True,
                        help="Whether or not to teleport model B for second interpolation")
    parser.add_argument("--same_cob", action="store_true", default=False,
                        help="Whether or not to use the same cob for the teleportation A and B")
    parser.add_argument("--cob_range", type=float, default=0.9,
                        help="Defines the range used for the COB. It must be a valid mix with cob_sampling")
    parser.add_argument("--cob_sampling", type=str, default="intra_landscape",
                        choices=['intra_landscape', 'inter_landscape', 'positive', 'negative', 'centered'],
                        help="Defines the type of sampling used for the COB. It must be a valid mix with cob_range")

    # Interpolation params
    parser.add_argument("--x", nargs=3, type=float, default=[-0.5, 1.5, 101], help="Defines the precision of the alpha")
    parser.add_argument("--model", type=str, default="MLPCOB", choices=get_model_names())
    parser.add_argument("--save_path", type=str, default='Interpolation', help='Path to save folder')
    parser.add_argument("--weightsA", type=str, help="Weights for model A", default=None)
    parser.add_argument("--weightsB", type=str, help="Weights for model B", default=None)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=['mnist', 'cifar10', 'cifar100'])

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    assert args.teleportA or args.teleportB, "At least one model must be teleported for second interpolation."
    if args.same_cob:
        assert args.teleportA and args.teleportB, "Both models must be teleported to use same_cob"

    save_path = get_nonexistent_path(args.save_path)

    with open(pjoin(save_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=0)

    device = 'cuda' if cuda_avail() else 'cpu'

    trainset, valset, testset = get_dataset_subsets(args.dataset)

    metric = TrainingMetrics(
        criterion=nn.CrossEntropyLoss(),
        metrics=[accuracy]
    )

    modelA = get_model(args.dataset, args.model, device=device)
    configA = LandscapeConfig(
        optimizer=(args.optimizerA, {"lr": args.lr, 'momentum': 0.9, 'weight_decay': 5e-4}),
        epochs=args.epochs,
        batch_size=args.batch_sizeA,
        cob_range=args.cob_range,
        cob_sampling=args.cob_sampling,
        teleport_at=[args.epochs],
        device=device
    )

    modelB = get_model(args.dataset, args.model, device=device)
    configB = LandscapeConfig(
        optimizer=(args.optimizerB, {"lr": args.lr, 'momentum': 0.9, 'weight_decay': 5e-4}),
        epochs=args.epochs,
        batch_size=args.batch_sizeB,
        cob_range=args.cob_range,
        cob_sampling=args.cob_sampling,
        teleport_at=[args.epochs],
        device=device
    )

    if args.weightsA is not None:
        modelA.load_state_dict(torch.load(args.weightsA))
        # modelA = torch.load(args.weightsA)
    if args.weightsB is not None:
        modelB.load_state_dict(torch.load(args.weightsB))
        # modelB = torch.load(args.weightsB)

    if args.train:
        print("Train model A")
        train(modelA, trainset, metric, configA, val_dataset=valset)
        print("Train model B")
        train(modelB, trainset, metric, configB, val_dataset=valset)

        torch.save(modelA.state_dict(), pjoin(save_path, 'modelA.pt'))
        torch.save(modelB.state_dict(), pjoin(save_path, 'modelB.pt'))

    res = test(modelA, valset, metric, configA)
    print("Model A Scored {} acc on valset".format(res['accuracy']))

    res = test(modelB, valset, metric, configB)
    print("Model B Scored {} acc on valset".format(res['accuracy']))

    a = torch.linspace(args.x[0], args.x[1], int(args.x[2]))

    teleportation_model = get_model(args.dataset, args.model, device=device)
    interpolation_config = LandscapeConfig(
        batch_size=1000,
        device=device
    )

    # interpolate between two original models
    print("Interpolating between original models...")
    param_o = modelA.get_params()
    param_t = modelB.get_params()

    loss, acc_t, loss_v, acc_v = generate_1D_linear_interp(teleportation_model, param_o, param_t, a, metric=metric,
                                                           config=interpolation_config, trainset=trainset,
                                                           valset=valset)

    res = {'train_loss': loss, 'train_acc': acc_t, 'val_loss': loss, 'val_acc': acc_v, 'alpha': a}
    torch.save(res, pjoin(save_path, 'interAB.pt'))

    plot_interp(loss, acc_t, a, acc_val=acc_v, loss_val=loss_v, title='Interpolation between model A and B',
                savepath=pjoin(save_path, 'interAB.jpg'))

    # Random teleportation
    if args.teleportA:
        print("Teleport model A")
        cob = modelA.generate_random_cob(args.cob_range, args.cob_sampling)
        modelA.teleport(cob)

    if args.teleportB:
        print("Teleport model B")
        if not args.same_cob:
            cob = modelB.generate_random_cob(args.cob_range, args.cob_sampling)
        modelB.teleport(cob)

    teleportation_model.set_params(*modelA.get_params())
    res = test(teleportation_model, valset, metric, configA)
    print("Model A Scored {} acc on valset".format(res['accuracy']))

    teleportation_model.set_params(*modelB.get_params())
    res = test(teleportation_model, valset, metric, configB)
    print("Model B Scored {} acc on valset".format(res['accuracy']))

    # interpolate between teleported models
    print("Interpolating between teleported models...")
    param_o = modelA.get_params()
    param_t = modelB.get_params()

    print("Before: ", loss)

    loss, acc_t, loss_v, acc_v = generate_1D_linear_interp(teleportation_model, param_o, param_t, a, metric=metric,
                                                           config=interpolation_config, trainset=trainset,
                                                           valset=valset)

    res = {'train_loss': loss, 'train_acc': acc_t, 'val_loss': loss_v, 'val_acc': acc_v, 'alpha': a}
    torch.save(res, pjoin(save_path, 'inter_TA_TB.pt'))

    print("After: ", loss)

    plot_interp(loss, acc_t, a, acc_val=acc_v, loss_val=loss_v,
                title='Interpolation between model {} and {} \n cob range: {}, {}{}'.format(
                    'T(A)' if args.teleportA else "A",
                    'T(B)' if args.teleportB else "B",
                    args.cob_range,
                    args.cob_sampling,
                    ", same cob" if args.same_cob else ""),
                savepath=pjoin(save_path, 'inter_TA_TB.jpg'))
