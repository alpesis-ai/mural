import numpy as np


def tokenize(text):
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {char: i for i, char in int2char.items()}
    encoded = np.array([char2int[char] for char in text])
    return encoded


def onehot_encode(array, n_labels):
    onehot = np.zeros((np.multiply(*array.shape), n_labels), dtype=np.float32)
    onehot[np.arange(onehot.shape[0]), array.flatten()] = 1.
    onehot = onehot.reshape(*array.shape, n_labels)
    return onehot


def get_batches(array, batch_size, seq_length):
    """
    A generator that returns batches of size with seq_length from a given array.
    """ 
    batch_size_total = batch_size * seq_length
    n_batches = len(array)//batch_size_total
    
    array = array[:n_batches * batch_size_total]
    array = array.reshape((batch_size, -1))
    
    for n in range(0, array.shape[1], seq_length):
        x = array[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, 0]
        yield x, y

