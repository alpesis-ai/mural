import numpy as np


def timesteps_generate(seq_length, step):
    return np.linspace(step * np.pi, (step+1) * np.pi, seq_length+1)


def timeseries_generate(timesteps):
    return np.sin(timesteps)
