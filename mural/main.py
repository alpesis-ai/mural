import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import settings
from train import train
from models.perceptrons import Perceptron
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

    return parser.parse_args()


if __name__ == '__main__':

    args = set_params()

    train_loader, test_loader = data_loader(args.dataset)
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    image_show(image[0, :]) 

    epochs = 2
    model = Perceptron()
    loss = nn.NLLLoss()

    if (args.optimizer == "ADAM"):
        optimizer = optim.Adam(model.parameters(), lr=0.003)
    elif (args.optimizer == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=0.003)
    else:
        print("Optimizer unknown.")
        exit(1)
    train(epochs, train_loader, model, loss, optimizer) 

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # calculate the class probabilities (softmax) for img
    probabilities = torch.exp(model(images[1]))

    if args.dataset == "MNIST":
        labels = settings.DATA_MNIST_LABELS 
    elif args.dataset == "FASHIONMNIST":
        labels = settings.DATA_FASHION_LABELS
    image_predict(images[1], probabilities, labels)
