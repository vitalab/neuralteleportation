if __name__ == '__main__':
    import operator

    from torch import nn

    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.config import TrainingMetrics
    from neuralteleportation.training.optim_gradient_score_training import train, TeleportationTrainingConfig
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.training.experiment_run import run_single_output_training
    from neuralteleportation.utils.gradient_eval import gradient_to_weight_norm

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    config = TeleportationTrainingConfig(input_shape=(3, 32, 32), device='cuda',
                                         comparison_metric=(gradient_to_weight_norm, operator.gt))
    run_single_output_training(train, get_cifar10_models(device='cuda'), config, metrics,
                               cifar10_train, cifar10_test, val_set=cifar10_val)
