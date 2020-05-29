import numpy as np
import torch


# def get_random_cob(range_cob: int, size: int, sampling_type='usual') -> np.ndarray:
#     """
#         Return random change of basis between -range_cob+1 and range_cob+1.
#         'usual' - in interval [1-range_cob,1+range_cob]
#         'symmetric' - equally in intervals [-1-range_cob,-1+range_cob] and [1-range_cob,1+range_cob]
#         'negative' - in interval [-1-range_cob,-1+range_cob]
#         'zero' - in interval [-range_cob,range_cob]
#
#     Args:
#         range_cob (int): range_cob for the change of basis. Recommended between 0 and 1, but can take any
#         positive range_cob.
#         size (int): size of the returned array.
#         sampling_type: label for type of sampling for change of basis
#     Returns:
#         ndarray of size size.
#     """
#     # Change of basis in interval [1-range_cob,1+range_cob]
#     if sampling_type == 'usual':
#         return np.random.uniform(low=-range_cob, high=range_cob, size=size).astype(np.float) + 1
#
#     # Change of basis in intervals [-1-range_cob,-1+range_cob] and [1-range_cob,1+range_cob]
#     elif sampling_type == 'symmetric':
#         samples = np.random.randint(0, 2, size=size)
#         cob = np.zeros_like(samples, dtype=np.float)
#         cob[samples == 1] = np.random.uniform(low=-1-range_cob, high=-1+range_cob, size=samples.sum())
#         cob[samples == 0] = np.random.uniform(low=1-range_cob, high=1+range_cob, size=(len(samples) - samples.sum()))
#         return cob
#
#     # Change of basis in interval [-1-range_cob,-1+range_cob]
#     elif sampling_type == 'negative':
#         return np.random.uniform(low=-range_cob, high=range_cob, size=size).astype(np.float) - 1
#
#     # Change of basis in interval [-range_cob,range_cob]
#     # This will produce very big weights in the network. Use only if needed.
#     elif sampling_type == 'zero':
#         return np.random.uniform(low=-range_cob, high=range_cob, size=size).astype(np.float)


def get_random_cob(range_cob: int, size: int, requires_grad=False, sampling_type='usual'):
    x = (-range_cob - range_cob) * torch.rand(size) + range_cob
    if requires_grad:
        x = x.requires_grad_()
    return x
#
# if __name__ == '__main__':
#     cob = get_random_cob(10, 10, True)
#     print(cob)