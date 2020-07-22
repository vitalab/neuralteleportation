"""
Perform gradient descent on a network's change of basis to find the change of basis that teleports
to a given set of weights.
"""

import argparse
import torch

from matplotlib import pyplot as plt
from torch import optim

from neuralteleportation.models.model_zoo.mlpcob import MLPCOB
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help='Number of optimization steps')
    parser.add_argument("--seed", type=int, default=1234, help='Seed for torch random')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate for cob optimizer')
    parser.add_argument("--cob_range", type=float, default=1,
                        help='Range for the teleportation to create target weights')

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    torch.manual_seed(args.seed)

    model = NeuralTeleportationModel(network=MLPCOB(input_shape=(1, 28, 28), num_classes=10),
                                     input_shape=(1, 1, 28, 28))

    # Get the initial set of weights and teleport.
    initial_weights = model.get_weights()
    model.random_teleport(cob_range=args.cob_range)

    # Get second set of weights (target weights)
    target_weights = model.get_weights()
    # Get the change of basis that created this set of weights.
    target_cob = model.get_cob(concat=True)

    # Generate a new random cob
    cob = model.generate_random_cob(cob_range=args.cob_range, requires_grad=True)

    history = []
    cob_error_history = []

    print("Initial error: ", (cob - target_cob).abs().mean().item())
    print("Target cob sample: ", target_cob[0:10].data)
    print("cob sample: ", cob[0:10].data)

    optimizer = optim.Adam([cob], lr=args.lr)

    """
    Optimize the cob to find the 'target_cob' that produced the original teleportation. 
    """
    for e in range(args.steps):
        # Reset the initial weights.
        model.set_weights(initial_weights)

        # Teleport with this cob
        model.teleport(cob)

        # Get the new weights and calculate the loss
        weights = model.get_weights()
        loss = (weights - target_weights).square().mean()

        # Backwards pass
        # add retain_graph=True to avoid error when running backward through the graph a second time
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        history.append(loss.item())
        cob_error_history.append((cob - target_cob).square().mean().item())
        if e % 100 == 0:
            print("Step: {}, loss: {}, cob error: {}".format(e, loss.item(), (cob - target_cob).abs().mean().item()))

    print("Final error: ", (cob - target_cob).abs().mean().item())
    print("Target cob: ", target_cob[0:10].data)
    print("cob: ", cob[0:10].data)

    print("Inital weights sample: ", initial_weights[:10])
    print("Target weights sample: ", target_weights[:10])
    print("Cob teleported weights sample: ", weights[:10])
    print("Target weights/teleported weights  diff ([:10]): ", (target_weights[:10] - weights[:10]).abs())
    print("Target weights/teleported weights  diff ([-10:]): ", (target_weights[-10:] - weights[-10:]).abs())
    print("Target weights/teleported weights diff mean: ", (target_weights - weights).abs().mean())

    plt.figure()
    plt.plot(history)
    plt.title("Loss")
    plt.show()

    plt.figure()
    plt.plot(cob_error_history)
    plt.title("Cob error")
    plt.show()
