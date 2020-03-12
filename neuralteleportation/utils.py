import numpy as np


def get_random_cob(range, size):
    samples = np.random.randint(0, 2, size=size)
    cob = np.zeros_like(samples, dtype=np.float)
    cob[samples == 1] = np.random.uniform(low=1 / range, high=1, size=samples.sum())
    cob[samples == 0] = np.random.uniform(low=1, high=range, size=(len(samples) - samples.sum()))

    return cob
