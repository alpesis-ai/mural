import settings
from text_predictor.text_processing import tokenize


def load_dummy():
    with open(settings.DATA_BOOKS_DIR + 'dummy.txt', 'r') as f:
        text = f.read()
    return text


def load_anna():
    with open(settings.DATA_BOOKS_DIR + 'anna.txt', 'r') as f:
        text = f.read()
    return text


def generate_data(train_data):
    encoded = tokenize(train_data)
    valid_idx = int(len(encoded)*(1-settings.DATA_VALID_SIZE))
    train_data = encoded[:valid_idx]
    valid_data = encoded[valid_idx:]

    return train_data, valid_data
