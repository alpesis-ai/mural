import argparse

import torch
from torch import nn
from torchvision import datasets, transforms

import settings
from train import train, train_with_steps
from test import test_with_steps
from models.perceptrons import Perceptron
from models.optimizers import define_optimizer
from processors.torchvision_datasets import data_loader
from visualizers.images import image_show, image_predict


def set_params():
    parser = argparse.ArgumentParser(description='Mural Parameters')

    parser.add_argument('--dataset',
                        type=str,
                        help="""Datasets: [MNIST, FASHIONMNIST]
                             """)

    parser.add_argument('--optimizer',
                        type=str,
                        help="""Optimizer: [ADAM, SGD]""")

    parser.add_argument('--epochs',
                        type=int,
                        help="epochs")

    parser.add_argument('--validation',
                        type=str,
                        help="validation: [SINGLE, STEPS]")

    return parser.parse_args()


def validate_single(epochs, train_loader, model, criterion, optimizer):
    train(epochs, train_loader, model, criterion, optimizer)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # calculate the class probabilities (softmax) for img
    probabilities = torch.exp(model(images[1]))

    if args.dataset == "MNIST":
        labels = settings.DATA_MNIST_LABELS 
    elif args.dataset == "FASHIONMNIST":
        labels = settings.DATA_FASHION_LABELS
    image_predict(images[1], probabilities, labels)

    

def validate_steps(epochs):
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



if __name__ == '__main__':

    args = set_params()

    train_loader, test_loader = data_loader(args.dataset)
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    image_show(image[0, :]) 

    model = Perceptron()
    criterion = nn.NLLLoss()
    optimizer = define_optimizer(args.optimizer, model)

    if (args.validation == "SINGLE"):
        validate_single(args.epochs, train_loader, model, criterion, optimizer)
    elif (args.validation == "STEPS"):
        validate_steps(args.epochs)
    else:
        print("validation error")
        exit(1)

