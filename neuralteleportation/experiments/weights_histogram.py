import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from neuralteleportation.training.experiment_setup import get_model, get_model_names


def argument_parser() -> argparse.Namespace:
    """
        Simple argument parser for the experiment.
    """
    parser = argparse.ArgumentParser(description='Simple argument parser for the histograms before and'
                                                 'after teleportation of a model.')

    # Model Parameters
    parser.add_argument("--model", "-m", type=str, default="MLPCOB", choices=get_model_names())
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"],
                        help="Dataset used to initialize the shape of the network")

    # Hyper Parameters
    parser.add_argument("--cob_range", type=float, default=0.9, help='set the CoB range for the teleportation.')
    parser.add_argument("--cob_sampling", type=str, default="intra_landscape", help="Sampling type for CoB.")

    # Plotting Parameters
    parser.add_argument("--plot_mode", type=str, default="modelwise", choices=["layerwise", "modelwise"],
                        help="Mode that determines how many histograms to produce: \n"
                             "- layerwise: Plot an histogram of the weights for each layer \n"
                             "- modelwise: Plot a single histogram of the weights that groups the whole network")
    parser.add_argument("--xlim", type=float, default=0.25,
                        help="Bound (on either side of 0) between which to display the histogram. "
                             "Only used for the histogram of teleported weights")
    parser.add_argument("--ylim_max", type=float, default=30,
                        help="Upper bound to use when displaying the histogram bars. Any values above this threshold "
                             "will be cropped")
    parser.add_argument("--output_dir", type=Path, default=Path.cwd(),
                        help="Root directory where to save the generated plots. "
                             "Only used if `save_format` is specified.")
    parser.add_argument("--save_format", type=str, choices=["png", "pdf"], default=None,
                        help="Format to use to save the generated plots. "
                             "If none is provided, plots will be displayed interactively rather than saved.")

    return parser.parse_args()


def plot_model_weights_histogram(model: NeuralTeleportationModel, mode: str, title: str, output_dir: Path = None,
                                 save_format: str = None, xlim: float = None, ylim_max: float = None,
                                 zoom_plot: bool = False) -> None:

    def _format_ticklabels(ticklabels) -> List[str]:
        return [f"{ticklabel:.1f}" for ticklabel in ticklabels]

    def _zoom_plot(ax, data, inset, lims):
        with sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': 'black'}):
            axins = ax.inset_axes(inset)
            sns.kdeplot(data, fill=True, ax=axins)
            axins.set_xlim(lims[:2])
            axins.set_ylim(lims[2:])
            axins.set(xticklabels=[], ylabel=None, yticklabels=[])
            rectangle_patch, connector_lines = ax.indicate_inset_zoom(axins)
            # Make the indicator and connectors more easily visible
            rectangle_patch.set_linewidth(2)
            for connector_line in connector_lines:
                connector_line.set_linewidth(2)
            # Manually set the visibility of the appropriate connector lines
            connector_lines[0].set_visible(True)  # Lower left
            connector_lines[1].set_visible(False)  # Upper left
            connector_lines[2].set_visible(True)  # Lower right
            connector_lines[3].set_visible(False)  # Upper right
            return axins

    def _plot_1d_array_histogram(array: np.ndarray, title: str):
        with sns.axes_style("darkgrid"):
            axes = sns.kdeplot(array, fill=True)
            if xlim:
                axes.set(xlim=(-xlim, xlim))
            if ylim_max:
                axes.set(ylim=(0, ylim_max))
            axes.set(ylabel=None, yticklabels=[])
            axes.set_xticklabels(_format_ticklabels(axes.get_xticks()), size=20)
            if zoom_plot:
                _zoom_plot(axes, array, inset=[0.05, 0.6, 0.35, 0.35], lims=[-0.13, -0.12, 0, 0.5])
                _zoom_plot(axes, array, inset=[0.6, 0.6, 0.35, 0.35], lims=[0.12, 0.13, 0, 0.5])
            if save_format:
                plt.savefig(output_dir / f"{title}.{save_format}", bbox_inches='tight')
                plt.close()
            else:
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
    from neuralteleportation.training.teleport.pseudo import DistributionTeleportationTrainingConfig, \
        simulate_teleportation_distribution
    from copy import deepcopy

    args = argument_parser()

    if args.save_format:    # Make sure the output directory exists, if we are to save the plots
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the model
    model_kwargs = {}
    if args.model == "MLPCOB":
        model_kwargs["hidden_layers"] = (500, 500, 500, 500, 500)
    net = get_model(args.dataset, args.model, **model_kwargs)

    # Plot an histogram of its weights
    plot_model_weights_histogram(net, args.plot_mode, title="kaiming_init",
                                 output_dir=args.output_dir, save_format=args.save_format,
                                 xlim=args.xlim, ylim_max=args.ylim_max)

    # Plot histogram of weights with same distribution
    # sampled from the cumulative distribution
    model = simulate_teleportation_distribution(deepcopy(net), config=DistributionTeleportationTrainingConfig())
    plot_model_weights_histogram(model, args.plot_mode, title="Weights with distribution simulated from teleportation",
                                 output_dir=args.output_dir, save_format=args.save_format,
                                 xlim=args.xlim, ylim_max=args.ylim_max)

    # Teleport the model and plot an histogram of its teleported weights
    net.random_teleport(args.cob_range, args.cob_sampling)
    plot_model_weights_histogram(net, args.plot_mode, title=f"Weights of teleported network "
                                                            f"with CoB range = {args.cob_range}",
                                 output_dir=args.output_dir, save_format=args.save_format,
                                 xlim=args.xlim, ylim_max=args.ylim_max, zoom_plot=True)


if __name__ == '__main__':
    main()
