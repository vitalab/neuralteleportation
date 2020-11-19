# How to generate mean plots

First of all make sure you have the .comet.config properly setup, 
then you can run the script from the directory containing the .comet.config as such:
```shell script
python generate_mean_graphs.py --metrics <list of metrics as seen on Comet.ml> --group_by <hyperparameters to group runs by> [--experiment_ids <list of Comet.ml experiment ids>|--experiment_dir <folder containing subfolders with the structure {experiment_id}/metrics.csv> ] --out_dir <path to directory where to write the plots>
```
A working example of the above command would be :
```shell script
python generate_mean_graphs.py --metrics train_loss validate_accuracy --group_by cob_range --experiment_ids 047fc65c4f7846bbb98270cff44fbf01 2be3094a069c48739ed2793bda4deb12 a43e6658e51f435b8e191ff5383a68cc b7501c677dcf46a3840c0cddf2607771 --out_dir ./mean_plots
```
This will generate two plots, one for `train_loss` and another for `validate_accuracy`, both plots would contain curves of runs grouped by `cob_range`.

For the plots to be complete and not trimmed due to comet's rate limiting, the experiments should have the `metrics.csv` saved as an asset.

For the `--group_by` option you can also pass in a list of hyperparameters like: `--group_by cob_range cob_sampling` which will group the runs with the same values for the passed hyperparameters.