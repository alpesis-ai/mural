import torch

import settings


def infer_single(feature, model_cls, batch_size, seq_length):
    model_cls.eval()
    hidden = model_cls.init_hidden(batch_size)
    #feature = feature.to(settings.DEVICE)
    output, hidden = model_cls(feature, hidden)
    prediction = torch.round(output.squeeze())
    print("Prediction value, pre-rounding: {:.6f}".format(output.item()))
    if (prediction.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected!")
