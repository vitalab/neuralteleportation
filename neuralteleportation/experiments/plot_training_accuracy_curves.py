import pathlib
import argparse

from neuralteleportation.utils.plot_hft5_training_curves import plot_training_curves


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder location containing training comparison .h5 files")
    parser.add_argument("--show_err", action="store_true", default=False,
                        help="Plot the data with the error bar shown on each point")
    parser.add_argument("--show_grid", action="store_true", default=False,
                        help="Activates the grid on plot")
    parser.add_argument("--std_err", action="store_true", default=False,
                        help="Use standard deviation as error margin")

    return parser.parse_args()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    args = argument_parser()
    path = pathlib.Path(args.folder) if args.folder else pathlib.Path().absolute() / "results/"

    fig = plot_training_curves(path, args.show_err, args.std_err, args.show_grid, show=False)

    plt.figure(fig.number)
    plt.show()
