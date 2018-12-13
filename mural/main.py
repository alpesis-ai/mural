import argparse

import torch
from torch import nn
from torchvision import datasets, transforms

from processors.datasets import define_dataset
from learning.validation import validate_single, validate_steps
from learning.inference import infer_single, infer_multi
from learning.optimizers import define_optimizer
from learning.models import define_model
from visualizers.images import image_show_single, image_show_multi


def set_params():
    parser = argparse.ArgumentParser(description='Mural Parameters')

    parser.add_argument('--dataset',
                        type=str,
                        help="""Dataset: [MNIST, FASHIONMNIST, CATSDOGS]
                             """)

    parser.add_argument('--model',
                        type=str,
                        help="Model: [CLASSIFIER, CLASSIFIER_DROPOUT, DENSENET_TRANS]")

    parser.add_argument('--optimizer',
                        type=str,
                        help="""Optimizer: [ADAM, SGD, ADAM_TRANS, SGD_TRANS]""")

    parser.add_argument('--epochs',
                        type=int,
                        help="epochs")

    parser.add_argument('--learning',
                        type=str,
                        help="learning: [VALID_SINGLE, VALID_STEPS, INFER_SINGLE, INFER_MULTI]")

    return parser.parse_args()


if __name__ == '__main__':

    args = set_params()

    train_loader, test_loader = define_dataset(args.dataset)
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    image_show_single(image[0, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_model(args.model)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = define_optimizer(args.optimizer, model)

    if (args.learning == "VALID_SINGLE"):
        validate_single(args.epochs, train_loader, test_loader, model, criterion, optimizer, args.dataset)
    elif (args.learning == "VALID_STEPS"):
        validate_steps(device, args.epochs, train_loader, test_loader, model, criterion, optimizer)
    elif (args.learning == "INFER_SINGLE"):
        infer_single(test_loader, model, args.dataset)
    elif (args.learning == "INFER_MULTI"):
        infer_multi()
    else:
        print("validation error")
        exit(1)
