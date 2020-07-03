from neuralteleportation.changeofbasisutils import get_available_cob_sampling_types
from neuralteleportation.utils.logger import VisdomLogger

if __name__ == '__main__':
    import operator

    from torch import nn

    from neuralteleportation.metrics import accuracy
    from neuralteleportation.training.config import TrainingMetrics
    from neuralteleportation.training.optim_score.training import train
    from neuralteleportation.training.optim_score.config import TeleportationTrainingConfig
    from neuralteleportation.training.experiment_setup import get_cifar10_models, get_cifar10_datasets
    from neuralteleportation.training.experiment_run import run_model_training
    from neuralteleportation.utils.model_eval import weighted_grad_norm, loss_lookahead_diff

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    # Run on CIFAR10
    cifar10_train, cifar10_val, cifar10_test = get_cifar10_datasets()
    cob_ranges = [0.7, 1.2]
    cob_samplings = get_available_cob_sampling_types()
    teleport_every_n_epochs = [1, 2, 5, 10]
    for sampling_type in cob_samplings:
        for cob_range in cob_ranges:
            for n in teleport_every_n_epochs:
                for comparison_metric in [(weighted_grad_norm, operator.gt),
                                          (loss_lookahead_diff, operator.gt)]:
                    for model in get_cifar10_models(device='cuda'):
                        env_name = "{}_{}_optim_teleport_{}_{}_every_{}".format(model.__class__.__name__,
                                                                                comparison_metric[0].__name__,
                                                                                sampling_type, cob_range, n)
                        print("Starting: ", env_name)
                        config = TeleportationTrainingConfig(
                            input_shape=(3, 32, 32),
                            device='cuda',
                            cob_range=cob_range,
                            cob_sampling=sampling_type,
                            teleport_every_n_epochs=n,
                            epochs=20,
                            exp_logger=VisdomLogger(env=env_name),
                            comparison_metric=comparison_metric
                        )
                        run_model_training(train, model,
                                           config, metrics,
                                           cifar10_train, cifar10_test, val_set=cifar10_val)
