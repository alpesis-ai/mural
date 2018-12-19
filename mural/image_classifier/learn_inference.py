import torch
import numpy as np

import settings
from common.managers.labels import define_labels
from common.visualizers.images import image_predict_single, image_predict_multi
from image_classifier.learn_test import test, test_multi


def infer_single(test_loader, model_cls, dataset):
    state_dict = torch.load(settings.WEIGHT_PATH + 'checkpoint.pth')
    model_cls.load_state_dict(state_dict)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    image = images[2]
    label = labels[2]
    image.view(1, 784)

    probabilities = test(image, model_cls)

    labels = define_labels(dataset)
    image_predict_single(image, probabilities, labels)


def infer_multi(test_loader, model_cls, loss_fn, dataset):
    state_dict = torch.load(settings.WEIGHT_PATH + 'checkpoint.pth')
    model_cls.load_state_dict(state_dict)
    predicted_probabilities, predicted_labels, test_loss, class_correct, class_total = test_multi(test_loader, model_cls, loss_fn, dataset)

    label_names = define_labels(dataset)
    for i in range(len(label_names)):
        if class_total[i] > 0:
            print("Test Accuracy of {:10s}: {:2f}% ({:2d}/{:2d})".format(
                   label_names[i],
                   100 * class_correct[i] / class_total[i],
                   int(np.sum(class_correct[i])), int(np.sum(class_total[i]))))
        else:
            print("Test Accuracy of {:10s}: N/A (no training examples)".format(labels[i]))

    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {:.6f}, Test Accuracy (Overall): {:.2f}% ({:2d}/{:2d})".format(
           test_loss,
           100 * np.sum(class_correct) / np.sum(class_total),
           int(np.sum(class_correct)), int(np.sum(class_total))))

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    image_predict_multi(images, predicted_labels, labels, label_names) 
