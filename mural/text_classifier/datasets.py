import torch
from torch.utils.data import TensorDataset, DataLoader

import settings
from text_classifier.features import features_padding


def features_generate(reviews_ints, seq_length):
    features = features_padding(reviews_ints, seq_length=seq_length)
    assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
    assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
    # print first 10 values of the first 30 batches
    print(features[:30,:10])
    return features


def get_loader(text2int, encoded_labels, seq_length):
    features = features_generate(text2int, seq_length)

    split_idx = int(len(features) * (1 - settings.DATA_VALID_SIZE))
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
    test_idx = int(len(remaining_x) * 0.5)
    valid_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    valid_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    print("Feature Shapes:")
    print("Train set: {}".format(train_x.shape),
          "Validation set: {}".format(valid_x.shape),
          "Test set: {}".format(test_x.shape))

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=settings.DATA_BATCH_SIZE)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=settings.DATA_BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=settings.DATA_BATCH_SIZE)
    return train_loader, valid_loader, test_loader
