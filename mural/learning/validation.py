import torch

import settings
from learning.train import train, train_with_steps
from learning.test import test_with_steps
from visualizers.images import image_predict


def validate_single(epochs, train_loader, test_loader, model, criterion, optimizer, dataset):
    train(epochs, train_loader, model, criterion, optimizer)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # calculate the class probabilities (softmax) for img
    probabilities = torch.exp(model(images[1]))

    if dataset == "MNIST":
        labels = settings.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = settings.DATA_FASHION_LABELS
    image_predict(images[1], probabilities, labels)


def validate_steps(epochs, train_loader, test_loader, model, criterion, optimizer):
    train_losses = []
    test_losses = []
    for e in range(epochs):
        running_loss = train_with_steps(train_loader, model, criterion, optimizer)
        test_loss, accuracy = test_with_steps(test_loader, model, criterion)

        this_train_loss = running_loss / len(train_loader)
        this_test_loss = test_loss / len(test_loader)
        train_losses.append(this_train_loss)
        test_losses.append(this_test_loss)
        print("Epoch: {}/{}..".format(e+1, epochs),
              "Training Loss: {:.3f}..".format(this_train_loss),
              "Test Loss: {:.3f}..".format(this_test_loss),
              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
