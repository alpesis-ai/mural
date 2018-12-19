import settings
from common.data.text import text_load
from text_predictor.text_processing import tokenize


def load_dummy():
    return text_load(settings.DATA_BOOKS_DIR + 'dummy.txt')


def load_anna():
    return text_load(settings.DATA_BOOKS_DIR + 'anna.txt')


def generate_data(train_data):
    encoded = tokenize(train_data)
    valid_idx = int(len(encoded)*(1-settings.DATA_VALID_SIZE))
    train_data = encoded[:valid_idx]
    valid_data = encoded[valid_idx:]

    return train_data, valid_data
