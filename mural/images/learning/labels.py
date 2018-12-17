import images.settings.classifier


def define_labels(dataset):
    if dataset == "MNIST":
        labels = images.settings.classifier.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = images.settings.classifier.DATA_FASHION_LABELS
    elif dataset == "CIFAR10":
        labels = images.settings.classifier.DATA_CIFAR10_LABELS
    elif dataset == "CATSDOGS":
        labels = images.settings.classifier.DATA_CATSDOGS_LABELS

    return labels
