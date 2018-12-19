import torch

import settings
from text_predictor.learn_test import test_single


def predict_sampling(model_cls, size, prime='The', topk=None):
    state_dict = torch.load(settings.WEIGHT_PATH + 'checkpoint.pth')
    model_cls.load_state_dict(state_dict)
    model_cls.eval()

    # run through the prime characters
    chars = [char for char in prime]
    hidden = model_cls.init_hidden(1)
    for char in prime:
        char, hidden = test_single(char, model_cls, hidden, topk)
    chars.append(char)

    # pass in the previous character and get a new one
    for i in range(size):
        char, hidden = test_single(chars[-1], model_cls, hidden, topk)
        chars.append(char)

    predicted = ''.join(chars)
    print(predicted)
    return predicted
