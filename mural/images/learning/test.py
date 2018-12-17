import torch
import numpy as np

import settings.common
from images.learning.labels import define_labels


def test(image, model_cls):
    model_cls.eval()

    with torch.no_grad():
        output = model_cls.forward(image)
    probabilities = torch.exp(output)
    return probabilities


def valid_with_steps(test_loader, model_cls, loss_fn):
    valid_loss = 0.0
    accuracy = 0.0   
 
    model_cls.eval()
    for images, labels in test_loader:
        images, labels = images.to(settings.common.DEVICE), labels.to(settings.common.DEVICE)
        log_probabilities = model_cls.forward(images)
        # valid_loss += loss_fn(log_probabilities, labels)
        loss = loss_fn(log_probabilities, labels)
        valid_loss += loss.item() * images.size(0)
        probabilities = torch.exp(log_probabilities)
        top_probability, top_class = probabilities.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    return valid_loss, accuracy


def test_multi(test_loader, model_cls, loss_fn, dataset):
    """For inference"""

    test_loss = 0.0
    labels_expected = define_labels(dataset)
    class_correct = list(0. for i in range(len(labels_expected)))
    class_total = list(0. for i in range(len(labels_expected)))

    model_cls.eval()
    for images, labels in test_loader:
        images, labels = images.to(settings.common.DEVICE), labels.to(settings.common.DEVICE)
        output = model_cls(images)
        loss = loss_fn(output, labels)
        test_loss += loss.item() * images.size(0)
        top_probabilities, top_classes = torch.max(output, 1)
        correct = np.squeeze(top_classes.eq(labels.data.view_as(top_classes)))
        for i in range(settings.common.DATA_BATCH_SIZE):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    return top_probabilities, top_classes, test_loss, class_correct, class_total
