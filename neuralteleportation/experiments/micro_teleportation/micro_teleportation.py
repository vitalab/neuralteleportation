from neuralteleportation.utils.micro_tp_utils import *
from neuralteleportation.training.config import TrainingMetrics


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)


if __name__ == '__main__':
    '''
    This script computes the histograms of angles between micro-teleportations of 4 models (MLP, VGG, ResNet and DenseNet)
    for 100 micro-teleportations on CIFAR-10 and random data (with input and output shape like CIFAR-10).
    '''
    from neuralteleportation.training.experiment_setup import *
    from neuralteleportation.metrics import accuracy
    from neuralteleportation.models.model_zoo.vggcob import vgg16COB
    from neuralteleportation.models.model_zoo.resnetcob import resnet18COB
    from neuralteleportation.models.model_zoo.densenetcob import densenet121COB
    from neuralteleportation.models.model_zoo.mlpcob import MLPCOB

    metrics = TrainingMetrics(nn.CrossEntropyLoss(), [accuracy])

    cifar10_train, cifar10_val, cifar10_test = get_dataset_subsets("cifar10")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_shape = (32, 3, 32, 32)

    # MLP

    model = MLPCOB(input_shape=(3, 32, 32), num_classes=10, hidden_layers=(128,))
    model = NeuralTeleportationModel(network=model, input_shape=input_shape)

    weights_init(model)

    micro_teleportation_dot_product(network=model, dataset=cifar10_train,
                                    network_descriptor='MLP',
                                    device=device, random_data=False)

    weights_init(model)

    micro_teleportation_dot_product(network=model, dataset=cifar10_train,
                                    network_descriptor='MLP',
                                    device=device, random_data=True)

    # VGG

    model = vgg16COB(num_classes=10).to(device=device)
    model = NeuralTeleportationModel(network=model, input_shape=input_shape)

    weights_init(model)

    micro_teleportation_dot_product(network=model, dataset=cifar10_train,
                                    network_descriptor='VGG',
                                    device=device, random_data=False)

    weights_init(model)

    micro_teleportation_dot_product(network=model, dataset=cifar10_train,
                                    network_descriptor='VGG',
                                    device=device, random_data=True)
