import torch
import numpy as np

import settings
from images.learning.train import train, train_with_steps
from images.learning.test import test, valid_with_steps
from images.visualizers.images import image_predict_single
from images.visualizers.eval import loss_compare


def validate_single(epochs, train_loader, valid_loader, model, criterion, optimizer, dataset):
    train(epochs, train_loader, model, criterion, optimizer)
    torch.save(model.state_dict(), settings.WEIGHT_PATH + 'checkpoint.pth')

    dataiter = iter(valid_loader)
    images, labels = dataiter.next()
    # calculate the class probabilities (softmax) for img
    # probabilities = torch.exp(model(images[1]))
    probabilities = test(images[1], model)

    if dataset == "MNIST":
        labels = settings.DATA_MNIST_LABELS
    elif dataset == "FASHIONMNIST":
        labels = settings.DATA_FASHION_LABELS
    image_predict_single(images[1], probabilities, labels)


def validate_steps(epochs, train_loader, valid_loader, model, criterion, optimizer):
    train_losses = []
    valid_losses = []
    valid_loss_min = np.Inf
    for e in range(epochs):
        running_loss = train_with_steps(train_loader, model, criterion, optimizer)
        valid_loss, accuracy = valid_with_steps(valid_loader, model, criterion)

        this_train_loss = running_loss / len(train_loader.dataset)
        this_valid_loss = valid_loss / len(valid_loader.dataset)
        train_losses.append(this_train_loss)
        valid_losses.append(this_valid_loss)
        print("Epoch: {}/{}..".format(e+1, epochs),
              "Training Loss: {:.6f}".format(this_train_loss),
              "Validation Loss: {:.6f}".format(this_valid_loss),
              "Test Accuracy: {:.6f}".format(accuracy/len(valid_loader.dataset)))

        # save model if validation loss has decreased
        if (this_valid_loss <= valid_loss_min):
            print("----> Validation loss decreased ({:.6f} -> {:.6f}), saving model...".format(
                   valid_loss_min, this_valid_loss))
            torch.save(model.state_dict(), settings.WEIGHT_PATH + 'checkpoint.pth')
            valid_loss_min = this_valid_loss

    loss_compare(train_losses, valid_losses, "Training Losses", "Validation Losses")
