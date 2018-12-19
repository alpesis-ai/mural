import os.path

import torch

#-------------------------------------------------------------------#
# PERFORMANCES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------#
# ROOT SETTINGS

# root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = PROJECT_ROOT

#-------------------------------------------------------------------#
# DATA PATHS

DATA_DIR = PROJECT_ROOT + "/_data"
# image classification
DATA_MNIST_DIR = DATA_DIR + "/mnist/"
DATA_FASHIONMNIST_DIR = DATA_DIR + "/fashion_mnist/"
DATA_CIFAR10_DIR = DATA_DIR + "/cifar10/"
DATA_CATSDOGS_DIR = DATA_DIR + "/cats_dogs/"
# image generation
DATA_STYLE_TRANSFER_DIR = DATA_DIR + "/style_transfer/"
# texts
DATA_BOOKS_DIR = DATA_DIR + "/books/"
DATA_SENTIMENT_DIR = DATA_DIR + "/sentiment_analysis/"

#-------------------------------------------------------------------#
# WEIGHTS PATHS

WEIGHT_PATH = PROJECT_ROOT + "/_weights/"

#-------------------------------------------------------------------#
# DATASET PERFORMANCE

# for data processor
DATA_NUM_WORKERS = 0
DATA_BATCH_SIZE = 20
# percentage of training set to use as validation
DATA_VALID_SIZE = 0.2

DATASETS = [
"MNIST",
"FASHIONMNIST",
"CIFAR10",
"CATSDOGS",
"BOOK_DUMMY",
"BOOK_ANNA"
]

#-------------------------------------------------------------------#
# MODELS

MODELS = [
"CLASSIFIER",
"CLASSIFIER_DROPOUT",
"MLP",
"CNN",
"DENSENET121_TRANS",
"VGG19_FEATURES",
"RNN",
"CHARRNN",
"SENTIMENTRNN"
]


OPTIMIZERS = [
"ADAM",
"SGD",
"ADAM_TRANS",
"SGD_TRANS"
]


LOSSES = [
"NLL",
"CROSSENTROPY",
"BCE"
]
