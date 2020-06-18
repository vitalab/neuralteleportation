import glob
import h5py
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(path: pathlib.Path, show_err: bool = False,
                         std_err: bool = False, show_grid: bool = False,
                         show: bool = True) -> plt.Figure:
    """
        Plot all the hft5 files found within the given path.

        path (pathlib.Path) this must be a folder path.
        show_err (bool): whether to show or not the error bars on each point
        std_err (bool): whether to use the standard deviation as the error
        show_grid (bool): Show grid on the plot.

        Return:
            the created fig
    """
    assert path.is_dir(), "Path must be a folder location"

    fig = plt.figure()
    for fpath in glob.glob(str(path / "*.h5"), recursive=True):
        plot_single_model_training_curves(fpath, fig)

    assert len(fig.get_axes()) > 0, "Nothing was drawn onto the graph! Verify the folder used for this script!"

    plt.title("Neural Teleportation Model Training Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if show_grid:
        plt.grid()
    plt.legend()

    if show:
        plt.show()

    return fig


def plot_single_model_training_curves(path: pathlib.Path, figure: plt.Figure = None, show_err: bool = False,
                                      std_err: bool = False, show_grid: bool = False, show: bool = True):
    """
        Plot a single hft5 file containing the training curves of the models comparison

        path (pathlib.Path) this must be a folder path.
        figure (pyplot.Figure), this is if the user want to put the printed result to an already existing pyplot figure.
        show_err (bool): whether to show or not the error bars on each point
        std_err (bool): whether to use the standard deviation as the error
        show_grid (bool): Show grid on the plot.

        Return:
            the created figure or the passed figure.
    """
    splits = pathlib.Path(path).name.split('_')

    # This preliminary check to not read files that does not fit
    # the format from teleportation_train_comparison script
    if len(splits) != 8:
        return
    name = splits[6]

    fig = figure
    if not figure:
        fig = plt.figure()
        plt.title("Neural Teleportation Model Training Comparison")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        if show_grid:
            plt.grid()
        if show:
            plt.show()
    else:
        fig = plt.figure(figure.number)

    f = h5py.File(path, "r")
    for k in f.keys():
        data = np.array(f[k])
        assert data.ndim == 2, "The open file does not contain the right data format or is "
        x = np.arange(data.shape[1])
        mean = data.mean(axis=0)
        if std_err:
            err = data.std(axis=0)
        else:
            err = -np.amin(data, axis=0) + mean, np.amax(data, axis=0) - mean
        label = name + "_" + k
        if show_err:
            plt.errorbar(x, mean, yerr=err, fmt='-o', ms=3, label=label, capsize=5.0)
        else:
            plt.plot(mean, '-o', ms=3, label=label, )
    f.close()

    return fig
