import torch
import torch.optim as optim


def train(model, criterion, train_dataset, val_dataset=None, optimizer=None, metrics=None, epochs=10, batch_size=32, device='cpu'):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        train_step(model, criterion, optimizer, train_loader, epoch, device=device)
        test(model, criterion, metrics, val_loader, device=device)


def train_step(model, criterion, optimizer, train_loader, epoch, device='cpu'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, criterion, metrics, test_loader, device='cpu'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            # if metrics is not None:
            #     metric_results = compute_metrics(metrics, y=target, y_hat=output)

    test_loss /= len(test_loader.dataset)

    return test_loss


def compute_metrics(metrics, y_hat, y, prefix='', to_tensor=True):
    results = {}
    for metric in metrics:
        m = metric(y_hat, y)
        if to_tensor:
            m = torch.tensor(m)
        results[prefix + metric.__name__] = m

    return results
