from neuralteleportation.changeofbasisutils import get_available_cob_sampling_types

if __name__ == '__main__':
    import argparse
    import operator

    from torch import nn

    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.config import TrainingMetrics
    from neuralteleportation.training.optim_score.training import train
    from neuralteleportation.training.optim_score.config import TeleportationTrainingConfig
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.training.experiment_run import run_single_output_training
    from neuralteleportation.utils.model_eval import weighted_grad_norm, loss_lookahead_diff

    parser = argparse.ArgumentParser(description="Network optimization experiment where gradient descent jumps between "
                                                 "teleportations, based on a criterion evaluating the gradient of "
                                                 "multiple teleportations")
    parser.add_argument("--cob_sampling", type=str, choices=get_available_cob_sampling_types(),
                        default='within_landscape',
                        help="Sampling algorithm to use when generating a change of basis")
    parser.add_argument("--cob_range", type=float, default=0.5)
    parser.add_argument("--optim", type=str, choices=['grad_norm', 'lookahead'], default='grad_norm',
                        help="Criterion used to select the optimal teleportation to train")
    args = parser.parse_args()

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    if args.optim == 'grad_norm':
        comparison_metric = (weighted_grad_norm, operator.gt)
    else:  # args.optim == 'lookahead':
        comparison_metric = (loss_lookahead_diff, operator.gt)

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    config = TeleportationTrainingConfig(input_shape=(3, 32, 32), device='cuda',
                                         cob_range=args.cob_range, cob_sampling=args.cob_sampling,
                                         comparison_metric=comparison_metric)
    run_single_output_training(train, get_cifar10_models(device='cuda'), config, metrics,
                               cifar10_train, cifar10_test, val_set=cifar10_val)
