from torch import nn

import settings


def define_loss(name):
    if name not in settings.LOSSES:
        print("Loss input error.")
        exit(1)

    if (name == "NLL"):
        # negative log likelihood loss
        loss = nn.NLLLoss()
    elif (name == "CROSSENTROPY"):
        loss = nn.CrossEntropyLoss()
    elif (name == "BCE"):
        loss = nn.BCELoss()

    return loss
