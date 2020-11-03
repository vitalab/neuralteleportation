import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

ap = argparse.ArgumentParser()
ap.add_argument('path', help='Path to <model_dataset> folder')
ap.add_argument('title', help='Figure title')
args = ap.parse_args()

epochs_sample = [5, 10, 20, 95]

path = Path(args.path)
prefix = path.name

# Define 4 types of experiments
experiments_def = [
    {'path': path / (prefix + '_SGD/tele'), 'optimizer': 'SGD', 'is_tele': True},
    {'path': path / (prefix + '_SGD/no_tele'), 'optimizer': 'SGD', 'is_tele': False},
    {'path': path / (prefix + '_SGD_Mom/tele'), 'optimizer': 'SGD+Momentum', 'is_tele': True},
    {'path': path / (prefix + '_SGD_Mom/no_tele'), 'optimizer': 'SGD+Momentum', 'is_tele': False}
]

# Load the data
runs = []
for exp_def in experiments_def:
    for run_path in exp_def['path'].glob('*/'):
        run_df = pd.read_csv(run_path / 'metrics.csv')
        run_df = run_df.iloc[epochs_sample]  # subsample
        run_df['Optimizer'] = exp_def['optimizer']
        run_df['Type'] = 'Teleported' if exp_def['is_tele'] else 'Standard'
        run_df['Epoch'] = run_df['step']
        run_df['Validation Accuracy'] = run_df['val_accuracy']
        runs.append(run_df.copy())
df = pd.concat(runs, ignore_index=True)

# Make and save the plot
g = seaborn.catplot(
    x="Optimizer",
    y="Validation Accuracy",
    hue="Type",
    hue_order=['Standard', 'Teleported'],
    col="Epoch",
    data=df,
    kind="violin",
    split=True,
    height=6,
    aspect=.4,
    inner="stick",
    palette={"Standard": "0.8", "Teleported": "orange"},
    bw=0.5  # bandwith, affects the smoothness
)
g.fig.suptitle(args.title)
plt.tight_layout(pad=3.0)
plt.savefig(prefix + '.png')
