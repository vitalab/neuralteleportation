import pandas as pd
import torch.optim as optim

from torch.nn.modules import Flatten
from neuralteleportation.layers.layer_utils import swap_model_modules_for_COB_modules
from collections import defaultdict
from micro_teleportation.micro_tp_utils import *


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
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from neuralteleportation.metrics import accuracy
    import torch.nn as nn

    trans = list()
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    mnist_train = MNIST('/tmp', train=True, download=True, transform=trans)
    mnist_val = MNIST('/tmp', train=False, download=True, transform=trans)
    mnist_test = MNIST('/tmp', train=False, download=True, transform=trans)

    cnn_model = torch.nn.Sequential(
        nn.Conv2d(1, 32, 5),
        nn.ReLU(),
        nn.Conv2d(32, 32, 5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3*3*64, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 10, bias=True)
    )

    cnn_model = swap_model_modules_for_COB_modules(cnn_model)

    optim = torch.optim.Adam(params=cnn_model.parameters(), lr=.01)
    metrics = [accuracy]
    loss = nn.CrossEntropyLoss()
    train(cnn_model, criterion=loss, train_dataset=mnist_train, val_dataset=mnist_val, optimizer=optim, metrics=metrics,
          epochs=1, device='cpu', batch_size=1)
    print(test(cnn_model, loss, metrics, mnist_test))

    dot_product(network=cnn_model, dataset=mnist_test, network_descriptor='CNN on MNIST')
