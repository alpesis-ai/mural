from torch import optim


def define_optimizer(name, model):
    if (name == "ADAM"):
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        return optimizer

    elif (name == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=0.003)
        return optimizer

    elif (name == "ADAM_TRANS"):
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
        return optimizer

    elif (name == "SGD_TRANS"):
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.003)
        return optimizer

    else:
        print("Optimizer Unknown")
        exit(1)
