import pathlib
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import argparse


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
    args = argument_parser()
    path = pathlib.Path(args.folder) if args.folder else pathlib.Path().absolute() / "results/"

    fig = plt.figure()

    for fpath in glob.glob(str(path / "*.h5"), recursive=True):
        splits = pathlib.Path(fpath).name.split('_')

        # This preliminary check to not read files that does not fit
        # the format from teleportation_train_comparison script
        if len(splits) != 8:
            continue
        name = splits[6]

        f = h5py.File(fpath, "r")
        for k in f.keys():
            data = np.array(f[k])
            assert data.ndim == 2, "The open file does not contain the right data format or is "
            x = np.arange(data.shape[1])
            mean = data.mean(axis=0)
            if args.std_err:
                err = data.std(axis=0)
            else:
                err = -np.amin(data, axis=0) + mean, np.amax(data, axis=0) - mean
            label = name + "_" + k
            if args.show_err:
                plt.errorbar(x, mean, yerr=err, fmt='-o', ms=3, label=label, capsize=5.0)
            else:
                plt.plot(mean, '-o', ms=3, label=label, )
        f.close()

    assert len(fig.get_axes()) > 0, "Nothing was drawn onto the graph! Verify the folder used for this script!"

    plt.title("Neural Teleportation Model Training Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if args.show_grid:
        plt.grid()
    plt.legend()

    plt.show()
