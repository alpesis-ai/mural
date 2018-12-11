import argparse

import torch
from torch import nn
from torchvision import datasets, transforms

from learning.validation import validate_single, validate_steps
from learning.inference import infer_single, infer_multi
from learning.optimizers import define_optimizer
from learning.models import define_model
from processors.torchvision_datasets import data_loader
from visualizers.images import image_show


def set_params():
    parser = argparse.ArgumentParser(description='Mural Parameters')

    parser.add_argument('--dataset',
                        type=str,
                        help="""Datasets: [MNIST, FASHIONMNIST]
                             """)

    parser.add_argument('--model',
                        type=str,
                        help="Model: [CLASSIFIER, CLASSIFIER_DROPOUT]")

    parser.add_argument('--optimizer',
                        type=str,
                        help="""Optimizer: [ADAM, SGD]""")

    parser.add_argument('--epochs',
                        type=int,
                        help="epochs")

    parser.add_argument('--learning',
                        type=str,
                        help="learning: [VALID_SINGLE, VALID_STEPS, INFER_SINGLE, INFER_MULTI]")

    return parser.parse_args()


if __name__ == '__main__':

    args = set_params()

    train_loader, test_loader = data_loader(args.dataset)
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    image_show(image[0, :]) 

    model = define_model(args.model)
    criterion = nn.NLLLoss()
    optimizer = define_optimizer(args.optimizer, model)

    if (args.learning == "VALID_SINGLE"):
        validate_single(args.epochs, train_loader, test_loader, model, criterion, optimizer, args.dataset)
    elif (args.learning == "VALID_STEPS"):
        validate_steps(args.epochs, train_loader, test_loader, model, criterion, optimizer)
    elif (args.learning == "INFER_SINGLE"):
        infer_single(test_loader, model, args.dataset)
    elif (args.learning == "INFER_MULTI"):
        infer_multi()
    else:
        print("validation error")
        exit(1)
