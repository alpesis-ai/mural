import settings.classifier


def define_labels(dataset):
    if dataset == "MNIST":
        labels = settings.classifier.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = settings.classifier.DATA_FASHION_LABELS
    elif dataset == "CIFAR10":
        labels = settings.classifier.DATA_CIFAR10_LABELS
    elif dataset == "CATSDOGS":
        labels = settings.classifier.DATA_CATSDOGS_LABELS

    return labels
