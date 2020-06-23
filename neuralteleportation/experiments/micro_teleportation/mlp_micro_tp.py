import torch.optim as optim

import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/neuralteleportation/')
sys.path.append('/content/drive/My Drive/Colab Notebooks/neuralteleportation/neuralteleportation/experiments')

from torch.nn.modules import Flatten
from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules
from collections import defaultdict
from utils.micro_tp_utils import *


def train(model, criterion, train_dataset, val_dataset=None, optimizer=None, metrics=None, epochs=10, batch_size=32,
          device='cpu'):
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        train_step(model, criterion, optimizer, train_loader, epoch, device=device)
        if val_dataset:
            val_res = test(model, criterion, metrics, val_dataset, device=device)
            print("Validation: {}".format(val_res))


def train_step(model, criterion, optimizer, train_loader, epoch, device='cpu'):

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, criterion, metrics, dataset, batch_size=32, device='cpu'):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = model.cuda()
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            results['loss'].append(criterion(output, target).item())

            if metrics is not None:
                batch_results = compute_metrics(metrics, y=target, y_hat=output, to_tensor=False)
                for k in batch_results.keys():
                    results[k].append(batch_results[k])

    results = pd.DataFrame(results)
    return dict(results.mean())


def compute_metrics(metrics, y_hat, y, prefix='', to_tensor=True):
    results = {}
    for metric in metrics:
        m = metric(y_hat, y)
        if to_tensor:
            m = torch.tensor(m)
        results[prefix + metric.__name__] = m
    return results


if __name__ == '__main__':
    from torchvision.datasets import MNIST, CIFAR10, CIFAR100
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    import torch.nn as nn

    trans = list()
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    train_set = MNIST('/tmp', train=True, download=True, transform=trans)
    val_set = MNIST('/tmp', train=False, download=True, transform=trans)
    test_set = MNIST('/tmp', train=False, download=True, transform=trans)

    mlp_model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(784, 12),
        nn.ReLU(),
        nn.Linear(12, 12),
        nn.ReLU(),
        nn.Linear(12, 10)
    )

    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    optim = torch.optim.SGD(params=mlp_model.parameters(), lr=.01)
    metrics = [accuracy]
    loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train(mlp_model, criterion=loss, train_dataset=train_set, val_dataset=val_set, optimizer=optim, metrics=metrics,
          epochs=1, device=device, batch_size=1)
    print(test(mlp_model, loss, metrics, test_set, device=device))

    dot_product(network=mlp_model, dataset=test_set, network_descriptor='MLP on MNIST', device=device)

    train_set = CIFAR10('/tmp', train=True, download=True, transform=trans)
    val_set = CIFAR10('/tmp', train=False, download=True, transform=trans)
    test_set = CIFAR10('/tmp', train=False, download=True, transform=trans)

    mlp_model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(3072, 1536),
        nn.ReLU(),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    if torch.cuda.is_available():
        mlp_model = mlp_model.cuda()
    else:
        mlp_model = mlp_model.cpu()

    train(mlp_model, criterion=loss, train_dataset=train_set, val_dataset=val_set, optimizer=optim, metrics=metrics,
          epochs=1, device=device, batch_size=1)
    print(test(mlp_model, loss, metrics, test_set, device=device))

    dot_product(network=mlp_model, dataset=test_set, network_descriptor='MLP on CIFAR10', device=device)

    train_set = CIFAR100('/tmp', train=True, download=True, transform=trans)
    val_set = CIFAR100('/tmp', train=False, download=True, transform=trans)
    test_set = CIFAR100('/tmp', train=False, download=True, transform=trans)

    mlp_model = torch.nn.Sequential(
        Flatten(),
        nn.Linear(3072, 1536),
        nn.ReLU(),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 100)
    )

    mlp_model = swap_model_modules_for_COB_modules(mlp_model)

    if torch.cuda.is_available():
        mlp_model = mlp_model.cuda()
    else:
        mlp_model = mlp_model.cpu()

    train(mlp_model, criterion=loss, train_dataset=train_set, val_dataset=val_set, optimizer=optim, metrics=metrics,
          epochs=1, device=device, batch_size=1)
    print(test(mlp_model, loss, metrics, test_set, device=device))

    dot_product(network=mlp_model, dataset=test_set, network_descriptor='MLP on CIFAR100', device=device)
