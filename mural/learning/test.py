import torch
import numpy as np

import settings


def test(image, model_cls):
    model_cls.eval()

    with torch.no_grad():
        output = model_cls.forward(image)
    probabilities = torch.exp(output)
    return probabilities


def test_with_steps(test_loader, model_cls, loss_fn, dataset):
    if dataset == "MNIST":
        labels_expected = settings.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels_expected = settings.DATA_FASHION_LABELS

    test_losses = 0.0
    class_correct = list(0. for i in range(len(labels_expected)))
    class_total = list(0. for i in range(len(labels_expected)))
    
    model_cls.eval()
    for images, labels in test_loader:
        images, labels = images.to(settings.DEVICE), labels.to(settings.DEVICE)
        # log_probabilities = model_cls.forward(images)
        # test_loss += loss_fn(log_probabilities, labels)
        # probabilities = torch.exp(log_probabilities)
        # top_probability, top_class = probabilities.topk(1, dim=1)
        # equals = top_class == labels.view(*top_class.shape)
        # accuracy += torch.mean(equals.type(torch.FloatTensor))
        output = model_cls(images)
        loss = loss_fn(output, labels)
        test_losses += loss.item() * images.size(0)
        top_probability, top_class = torch.max(output, 1)
        correct = np.squeeze(top_class.eq(labels.data.view_as(top_class)))
        for i in range(settings.DATA_BATCH_SIZE):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    return test_losses, class_correct, class_total
