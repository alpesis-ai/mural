import argparse

import torch
from torch import nn
from torchvision import datasets, transforms

from learning.validation import validate_single, validate_steps
from models.perceptrons import Perceptron
from models.optimizers import define_optimizer
from processors.torchvision_datasets import data_loader
from visualizers.images import image_show


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
        validate_single(args.epochs, train_loader, test_loader, model, criterion, optimizer, args.dataset)
    elif (args.validation == "STEPS"):
        validate_steps(args.epochs, train_loader, test_loader, model, criterion, optimizer)
    else:
        print("validation error")
        exit(1)
