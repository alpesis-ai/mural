import settings


def define_labels(dataset):
    if dataset == "MNIST":
        labels = settings.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = settings.DATA_FASHION_LABELS
    elif dataset == "CIFAR10":
        labels = settings.DATA_CIFAR10_LABELS

    return labels
