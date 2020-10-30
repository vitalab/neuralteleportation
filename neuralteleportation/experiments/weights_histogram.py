import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.experiment_setup import get_model, get_model_names


def argument_parser() -> argparse.Namespace:
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the plot loss landscape experiment.')

    # Model Parameters
    parser.add_argument("--model", "-m", type=str, default="MLPCOB", choices=get_model_names())
    parser.add_argument("--dataset", type=str, default="mnist", choices=["cifar10", "mnist"],
                        help="Dataset used to initialize the shape of the network")

    # Hyper Parameters
    parser.add_argument("--cob_range", type=float, default=0.99, help='set the CoB range for the teleportation.')
    parser.add_argument("--cob_sampling", type=str, default="within_landscape", help="Sampling type for CoB.")

    # Plotting Parameters
    parser.add_argument("--plot_mode", type=str, default="layerwise", choices=["layerwise", "modelwise"],
                        help="Mode that determines how many histograms to produce: \n"
                             "- layerwise: Plot an histogram of the weights for each layer \n"
                             "- modelwise: Plot a single histogram of the weights that groups the whole network")
    parser.add_argument("--xlim", type=float, default=1.,
                        help="Bound (on either side of 0) between which to display the histogram. "
                             "Only used for the histogram of teleported weights")
    parser.add_argument("--ylim_max", type=float, default=16_000,
                        help="Upper bound to use when displaying the histogram bars. Any values above this threshold "
                             "will be cropped")

    return parser.parse_args()


def plot_model_weights_histogram(model: NeuralTeleportationModel, mode: str, title: str,
                                 xlim: float = None, ylim_max: float = None) -> None:

    def _plot_1d_array_histogram(array: np.ndarray, title: str):
        axes = sns.histplot(array, kde=True)
        axes.set_title(title)
        if xlim:
            axes.set(xlim=(-xlim, xlim))
        if ylim_max:
            axes.set(ylim=(0, ylim_max))
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
    model_kwargs = {}
    if args.model == "MLPCOB":
        model_kwargs["hidden_layers"] = (128,128,128,128)
    net = get_model(args.dataset, args.model, **model_kwargs)

    # Plot an histogram of its weights
    plot_model_weights_histogram(net, args.plot_mode, title="xavier_init", xlim=args.xlim, ylim_max=args.ylim_max)

    # Teleport the model and plot an histogram of its teleported weights
    net.random_teleport(args.cob_range, args.cob_sampling)
    plot_model_weights_histogram(net, args.plot_mode, title=f"{args.cob_range}_{args.cob_sampling}_cob",
                                 xlim=args.xlim, ylim_max=args.ylim_max)


if __name__ == '__main__':
    main()
