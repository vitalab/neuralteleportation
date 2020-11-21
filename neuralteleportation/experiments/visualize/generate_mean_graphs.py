import os
from collections import defaultdict
from io import StringIO
from glob import glob
from pathlib import Path
import yaml

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from tqdm import tqdm


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    Used to add spacing between box groups.
    """
    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for i, c in enumerate(ax.get_children()):
            # searching for PathPatches
            if isinstance(c, PathPatch):
                if i%2 != 0:
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5*(xmin+xmax)
                    xhalf = 0.5*(xmax - xmin)

                    # setting new width of box
                    # xmin_new = xmid-fac*xhalf
                    xmin_new = xmin
                    xmax_new = xmid+fac*xhalf
                    # xmax_new = xmax
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def shim_metric_name(metric_name):
    # This is used to normalize the metrics names between Comet and CSV
    name_map = {
        "val_loss": "validate_loss",
        "val_accuracy": "validate_accuracy",
        "val_accuracy_top5": "validate_accuracy_top5",
    }
    return name_map[metric_name] if metric_name in name_map.keys() else metric_name

def get_csv_from_comet(exp):
    asset_id = [asset["assetId"] for asset in exp.get_asset_list() if asset["fileName"] == "metrics.csv"]
    if len(asset_id) == 0:
        return None
    asset_content = exp.get_asset(asset_id, return_type="text")
    buff = StringIO(asset_content)
    return buff

def get_metrics(exp, csv_file_path=None):
    # Fetches CSV file from comet and generates cometAPI style metrics list,
    # if no csv file is found on comet, fallsback on the comet metrics.
    buff = csv_file_path
    if buff is None or not is_non_zero_file(buff):
        buff = get_csv_from_comet(exp)
        if buff is None:
            print(f"WARN: metrics.csv not found as a comet asset in {exp.id}, falling back on comet metrics.")
            return exp.get_metrics()
    df = pd.read_csv(buff)
    if len(df) < 20:
        epochs = np.arange(120)
        df[epochs, 'validate_accuracy'] = 50.0
        print("WARNING: FAKE TEST DATA INSERTED because less than 20 epochs")
    metrics = []
    for _, row in df.iterrows():
        r_keys = [k for k in row.keys() if k != "step"]
        for k in r_keys:
            metric_obj = {
                "epoch": int(row["step"]),
                "metricName": shim_metric_name(k),
                "metricValue": row[k],
            }
            metrics.append(metric_obj)
    return metrics


def fetch_data(exps_ids, metrics_filter, group_by, experiments_csv_dict=None):
    all_metrics_grouped = defaultdict(list)
    for exp_id in tqdm(exps_ids, desc="Fetching data: "):
        # Get hparams
        metrics_file = Path(experiments_csv_dict[exp_id])
        hparams_file = metrics_file.parent / 'hparams.yml'
        with open(hparams_file, 'r') as f:
            hparams = yaml.safe_load(f)

        # Get group name (e.g "SGD & no_teleport")
        def get_value(param_value):
            # For case where value is [class_name, constructor_params]
            if type(param_value) is list:
                v = param_value[0]
            else:
                v = param_value
            assert type(v) is str
            return v
        group_param_values = [get_value(v) for k, v in hparams.items() if k in group_by]
        assert len(group_param_values) > 0, f"ERROR: Experiment {exp_id} does not have any of the hyperparameters {group_by}!"
        group_name = " & ".join(group_param_values)

        # Get metrics array
        metrics = get_metrics(None, csv_file_path=experiments_csv_dict[exp_id])
        metrics = [metric for metric in metrics if metric["metricName"].lower() in metrics_filter]
        metrics_dict = {}
        for metric in metrics:
            name = metric["metricName"]
            value = metric["metricValue"]
            epoch = metric["epoch"]
            if name not in metrics_dict.keys():
                metrics_dict[name] = {epoch: value}
            else:
                metrics_dict[name][epoch] = value
        all_metrics_grouped[group_name].append(metrics_dict)
    return all_metrics_grouped


def find_best_legend_pos(metric_name):
    metric_name = metric_name.lower()
    if "loss" in metric_name:
        return "upper right"
    if "accuracy" in metric_name:
        return "lower right"
    return "best"


def plot_mean_std_curve(metrics_grouped, metric_name, group_by, output_dir, legend_pos="best"):
    # Upper bound on epochs, will be trimmed to max epochs
    n_epochs = 1000
    grouped_plots = {}
    for g_name in metrics_grouped.keys():
        g_metrics = metrics_grouped[g_name]
        nb_runs = len(g_metrics)
        g_data = np.zeros((n_epochs, nb_runs), dtype=np.float32)
        max_epochs = 0
        for i in range(nb_runs):
            run_metric = g_metrics[i][metric_name]
            for k in run_metric.keys():
                g_data[k, i] = float(run_metric[k])
                max_epochs = max(k, max_epochs)
        g_data = g_data[:max_epochs, :]
        grouped_plots[g_name] = {
            "data": g_data,
            "mean": np.mean(g_data, axis=1),
            "std": np.std(g_data, axis=1),
            "max_epochs": max_epochs
        }
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        fig.suptitle(f"{metric_name}")
        group_by_str = "_".join(group_by)
        output_filename = os.path.join(output_dir, f"{metric_name}_{group_by_str}.pdf")
        clrs = sns.color_palette("husl", len(list(grouped_plots.keys())))
        sort_labels = []
        for i, g_name in enumerate(sorted(list(grouped_plots.keys()))):
            g_mean = grouped_plots[g_name]["mean"]
            # Find first nan in the mean array
            not_nan_idx = np.argwhere(np.isnan(g_mean))
            not_nan_idx = not_nan_idx[0, 0] if not_nan_idx.size > 0 else -1
            # truncate the mean and std at the first nan found
            g_mean = g_mean[:not_nan_idx]
            g_std = grouped_plots[g_name]["std"]
            g_std = g_std[:not_nan_idx]
            epochs = range(min(grouped_plots[g_name]["max_epochs"], g_mean.shape[0]))
            # save the last valid val to sort the labels later
            sort_labels.append({"last_val": g_mean[-1], "label": g_name})
            ax.plot(epochs, g_mean, label=g_name, c=clrs[i])
            ax.fill_between(epochs, g_mean - g_std, g_mean + g_std, alpha=0.3, facecolor=clrs[i])
            ax.set_ylabel(f"{metric_name}")
            ax.set_xlabel("epoch")
        leg_pos = find_best_legend_pos(metric_name) if legend_pos == "infer" else legend_pos
        sort_labels = sorted(sort_labels, key=lambda dk: dk["last_val"], reverse=True)
        handles, labels = ax.get_legend_handles_labels()
        # Sort pyplot's legend with the sorted values
        handles = [handles[labels.index(dict_lbl["label"])] for dict_lbl in sort_labels]
        labels = [labels[labels.index(dict_lbl["label"])] for dict_lbl in sort_labels]
        ax.legend(handles, labels, loc=leg_pos)
        plt.savefig(output_filename)
        print(f"Saved plot at : {output_filename}")


def prettify_param_name(param):
    name_map = {
        "sgd & no_teleport": "Standard SGD",
        "sgd & teleport": "Teleported SGD",
        "sgd with momentum & no_teleport": "Standard SGD with momentum",
        "sgd with momentum & teleport": "Teleported SGD with momentum",
    }
    if param in name_map.keys():
        return name_map[param]
    return param

def prettify_metric_name(metric):
    name_map = {
        "validate_accuracy": "Validation Accuracy",
        "train_loss": "Training Loss",
    }
    if metric in name_map.keys():
        return name_map[metric]
    return metric

def plot_box(metrics_grouped, metric_name, epochs_sample, group_by, output_dir, legend_pos="best"):
    # Upper bound on epochs, will be trimmed to max epochs
    metric_human_name = prettify_metric_name(metric_name)
    n_epochs = 1000
    group_runs = []
    for g_name in metrics_grouped.keys():
        g_metrics = metrics_grouped[g_name]
        nb_runs = len(g_metrics)
        # Parses all runs
        for i in range(nb_runs):
            g_data = np.zeros((n_epochs,), dtype=np.float32)
            run_metric = g_metrics[i][metric_name]
            # Flatten metric data for this run
            for k in run_metric.keys():
                g_data[k] = float(run_metric[k])
            # DataFrame for plotting
            df = pd.DataFrame.from_dict({
                metric_human_name: g_data,
                "epoch": np.array(range(g_data.shape[0])),
                "hparam": prettify_param_name(g_name.lower()),
                "metric": metric_name,
            })
            # Subsample the dataframe
            df = df.iloc[epochs_sample]
            group_runs.append(df.copy())
    group_df = pd.concat(group_runs, ignore_index=True)
    with sns.axes_style("darkgrid"):
        group_by_str = "_".join(group_by)
        output_filename = os.path.join(output_dir, f"box_{metric_name}_{group_by_str}.pdf")
        clrs = sns.color_palette("Paired", 6)
        g = sns.catplot(
            x="epoch",
            y=metric_human_name,
            hue="hparam",
            # Hard Coded hue order for now
            hue_order=["Standard SGD", "Teleported SGD", "Standard SGD with momentum", "Teleported SGD with momentum"],
            data=group_df,
            height=6,
            aspect=1.4,
            palette={
                "Standard SGD": clrs[4],
                "Teleported SGD": clrs[5],
                "Standard SGD with momentum": clrs[2],
                "Teleported SGD with momentum": clrs[3]
            },
            kind="box",
            legend=False
        )
        plt.legend(loc=legend_pos)
        g.fig.suptitle(f"{metric_human_name}")
        g.set_xlabels("Epoch", fontsize=20)
        g.set_ylabels(metric_human_name, fontsize=20)
        g.fig.tight_layout(pad=3.0)
        adjust_box_widths(g.fig, 0.7)
        plt.savefig(output_filename)
        print(f"Saved box plot at : {output_filename}")


def parse_experiments_dir(experiment_dir):
    # Parser directory into experiment ids and their metrics.csv path
    all_dirs = glob(os.path.join(experiment_dir, "**/*.csv"), recursive=True)
    all_dirs = [p for p in all_dirs if "metrics.csv" in p.lower()]
    experiments_dict = {}
    for p in all_dirs:
        experiment_id = os.path.basename(os.path.dirname(p))
        experiments_dict[experiment_id] = p
    return experiments_dict, list(experiments_dict.keys())

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment_ids",
        type=str,
        nargs="+",
        help="IDs of the experiments to plot",
        default=None
    )
    group.add_argument(
        "--experiment_dir",
        type=str,
        help="Folder containing metrics.csv for each experiment",
        default=None
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Metrics to plot curves for.",
        required=True
    )
    parser.add_argument(
        "--group_by",
        type=str,
        nargs="+",
        help="Hyperparameter to group experiments with.",
        required=True
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Output directory to save figures in.",
        required=True
    )
    parser.add_argument(
        "--legend_pos",
        type=str,
        help="Position of the legend on the plots, (by default uses 'best' option of matplotlib)",
        default="best",
    )
    parser.add_argument(
        "--boxplot",
        action="store_true",
        help="Whether to plot the Boxplot.",
    )
    parser.add_argument(
        "--box_epochs",
        type=int,
        nargs="+",
        help="Epochs to subsample the box plot X axis with.",
        default=None
    )
    args = parser.parse_args()

    experiments_csv_dict = None
    experiments_ids = args.experiment_ids
    if args.experiment_ids:
        if len(args.experiment_ids) <= 1:
            print("ERROR: can't generate plots for one experiment!")
            exit(1)

    if args.experiment_dir:
        if not os.path.exists(args.experiment_dir):
            print(f"ERROR: {args.experiment_dir} does not exist!")
            exit(1)
        experiments_csv_dict, experiments_ids = parse_experiments_dir(args.experiment_dir)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    all_metrics_grouped = fetch_data(experiments_ids, args.metrics, args.group_by,
                                     experiments_csv_dict=experiments_csv_dict)
    for metric in args.metrics:
        plot_mean_std_curve(all_metrics_grouped, metric, args.group_by, args.out_dir, args.legend_pos)
        if args.boxplot:
            epochs_subsample = [5, 10, 20, 95] if args.box_epochs is None else args.box_epochs
            plot_box(all_metrics_grouped, metric, epochs_subsample, args.group_by, args.out_dir, args.legend_pos)
