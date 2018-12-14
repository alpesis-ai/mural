import argparse

import torch
from torchvision import datasets, transforms

import settings
from processors.datasets import define_dataset
from learning.validation import validate_single, validate_steps
from learning.inference import infer_single, infer_multi
from learning.losses import define_loss
from learning.optimizers import define_optimizer
from learning.models import define_model
from visualizers.images import preshow_images


def set_params():
    parser = argparse.ArgumentParser(description='Mural Parameters')

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help="""Dataset: [MNIST, FASHIONMNIST, CIFAR10, CATSDOGS]
                             """)

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help="Model: [CLASSIFIER, CLASSIFIER_DROPOUT, DENSENET_TRANS]")

    parser.add_argument('--loss',
                        type=str,
                        required=True,
                        help="Loss: [NLL, CROSSENTROPY]")

    parser.add_argument('--optimizer',
                        type=str,
                        help="""Optimizer: [ADAM, SGD, ADAM_TRANS, SGD_TRANS]""")

    parser.add_argument('--epochs',
                        type=int,
                        help="epochs (train only)")

    parser.add_argument('--learning',
                        type=str,
                        required=True,
                        help="learning: [VALID_SINGLE, VALID_STEPS, INFER_SINGLE, INFER_MULTI]")

    return parser.parse_args()


if __name__ == '__main__':

    args = set_params()

    train_loader, valid_loader, test_loader = define_dataset(args.dataset)

    if "VALID_" in args.learning:
        preshow_images(train_loader, args.dataset)
    elif "INFER_" in args.learning:
        preshow_images(test_loader, args.dataset)

    model = define_model(args.model)
    model.to(settings.DEVICE)

    criterion = define_loss(args.loss)
    if "VALID_" in args.learning:
        optimizer = define_optimizer(args.optimizer, model)

    if (args.learning == "VALID_SINGLE"):
        validate_single(args.epochs, train_loader, valid_loader, model, criterion, optimizer, args.dataset)
    elif (args.learning == "VALID_STEPS"):
        validate_steps(args.epochs, train_loader, valid_loader, model, criterion, optimizer)
    elif (args.learning == "INFER_SINGLE"):
        infer_single(test_loader, model, args.dataset)
    elif (args.learning == "INFER_MULTI"):
        infer_multi(test_loader, model, criterion, args.dataset)
    else:
        print("validation error")
        exit(1)
