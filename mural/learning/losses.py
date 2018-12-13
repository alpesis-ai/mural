from torch import nn


def define_loss(name):
    if (name == "NLL"):
        # negative log likelihood loss
        loss = nn.NLLLoss()
        return loss

    elif (name == "CROSSENTROPY"):
        loss = nn.CrossEntropyLoss()
        return loss

    else:
        print("Loss input error")
        exit(1)

