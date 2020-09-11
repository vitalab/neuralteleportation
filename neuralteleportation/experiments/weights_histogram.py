import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.experiment_setup import get_model


def argument_parser():
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the plot loss landscape experiment.')

    # Hyper Parameters
    parser.add_argument("--cob_range", type=float, default=0.99, help='set the CoB range for the teleportation.')
    parser.add_argument("--cob_sampling", type=str, default="within_landscape", help="Sampling type for CoB.")

    # Plotting Parameters
    parser.add_argument("--xlim", type=float, default=1.,
                        help="Bound (on either side of 0) between which to display the histogram. "
                             "Only used for the histogram of teleported weights")

    return parser.parse_args()


def plot_model_weights_histogram(model: NeuralTeleportationModel, mode: str = "layerwise", title: str = None,
                                 xlim: float = None):

    def _plot_1d_array_histogram(array: np.ndarray, title: str):
        axes = sns.histplot(array, kde=True)
        axes.set_title(title)
        if xlim:
            axes.set(xlim=(-xlim, xlim))
        axes.set_xlabel("weights_value")
        axes.set_ylabel("count")
        plt.show()

    print(f"Plotting {title} histogram ...")
    if mode == "modelwise":
        _plot_1d_array_histogram(model.get_weights().cpu().detach().numpy(), title)
    elif mode == "layerwise":
        # Get weights of each layer, without the bias weights
        layers = model.get_weights(concat=False, bias=False)

        # Get rid of layers connected to input and output nodes, to keep only hidden layers
        layers = layers[1:-1]

        for idx, layer_weights in enumerate(layers):
            _plot_1d_array_histogram(layer_weights.cpu().detach().numpy(), title + f"_layer{idx}")
    else:
        raise ValueError(f"Mode {mode} is not a valid option. Choose one of: {{modelwise,layerwise}}")


def main():
    args = argument_parser()

    # Initialize the model
    net = get_model("mnist", "MLPCOB", hidden_layers=(128,128,128,128))

    # Plot an histogram of its weights
    plot_model_weights_histogram(net, title="xavier_init")

    # Teleport the model and plot an histogram of its teleported weights
    net.random_teleport(args.cob_range, args.cob_sampling)
    plot_model_weights_histogram(net, title=f"{args.cob_range}_{args.cob_sampling}_cob", xlim=args.xlim)


if __name__ == '__main__':
    main()
