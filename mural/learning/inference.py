import torch

import settings
from learning.test import test
from visualizers.images import image_show, image_predict


def infer_single(test_loader, model_cls, dataset):
    state_dict = torch.load(settings.WEIGHT_PATH + 'checkpoint.pth')
    model_cls.load_state_dict(state_dict)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    image = images[2]
    label = labels[2]
    image.view(1, 784)

    probabilities = test(image, model_cls)

    if dataset == "MNIST":
        labels = settings.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = settings.DATA_FASHION_LABELS
    image_predict(image, probabilities, labels)


def infer_multi():
    pass
