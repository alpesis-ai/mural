from torch import optim

import settings


def define_optimizer_classifier(name, rate, model):
    if name not in settings.OPTIMIZERS:
        print("Optimizer unknown")
        exit(1)

    if (name == "ADAM"):
        optimizer = optim.Adam(model.parameters(), lr=rate)
    elif (name == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=rate)
    elif (name == "ADAM_TRANS"):
        optimizer = optim.Adam(model.classifier.parameters(), lr=rate)
    elif (name == "SGD_TRANS"):
        optimizer = optim.SGD(model.classifier.parameters(), lr=rate)

    return optimizer 


def define_optimizer_generator(name, rate, values):
    if name not in settings.OPTIMIZERS:
        print("Optimizer unknown")
        exit(1)

    if (name == "ADAM"):
        optimizer = optim.Adam(values, lr=rate)

    return optimizer
