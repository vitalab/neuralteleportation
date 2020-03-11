import torch.nn as nn
import torch


class LinearCOB(nn.Linear):

    def apply_cob(self, prev_cob, next_cob):
        w = torch.tensor(next_cob[..., None] / prev_cob[None, ...], dtype=torch.float32)
        self.weight = nn.Parameter(self.weight * w, requires_grad=True)

        if self.bias:
            self.bias = torch.nn.Parameter(self.bias * torch.tensor(next_cob, dtype=torch.float32), requires_grad=True)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
