import numpy as np

import settings


def select_data_single(data_loader):
    image, label = next(iter(data_loader))
    return image, label


def select_data_multi(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    return images, labels


def split_dataset(num_data):
    indices = list(range(num_data))
    split = int(np.floor(settings.DATA_VALID_SIZE * num_data))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx
