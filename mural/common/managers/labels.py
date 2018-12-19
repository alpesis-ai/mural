import common.data.labels


def define_labels(dataset):
    if dataset == "MNIST":
        labels = common.data.labels.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = common.data.labels.DATA_FASHION_LABELS
    elif dataset == "CIFAR10":
        labels = common.data.labels.DATA_CIFAR10_LABELS
    elif dataset == "CATSDOGS":
        labels = common.data.labels.DATA_CATSDOGS_LABELS

    return labels
