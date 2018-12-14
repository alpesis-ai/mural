import os.path

import torch

#-------------------------------------------------------------------#
# PERFORMANCES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for data processor
DATA_NUM_WORKERS = 0
DATA_BATCH_SIZE = 20
# percentage of training set to use as validation
DATA_VALID_SIZE = 0.2

DATASETS = [
"MNIST",
"FASHIONMNIST",
"CIFAR10",
"CATSDOGS"
]

#-------------------------------------------------------------------#
# Training

LEARNING_RATE = 0.01

MODELS = [
"CLASSIFIER",
"CLASSIFIER_DROPOUT",
"MLP",
"CNN",
"DENSENET121_TRANS"
]

#-------------------------------------------------------------------#
# VISUALIZATION

# image shown:
# - 0: not shown
# - 1: show a single image
# - 2: show multi images
# - 3: show image detail
IMAGE_EXPLORE = 2

#-------------------------------------------------------------------#
# ROOT SETTINGS

# root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#-------------------------------------------------------------------#
# DATA PATHS

DATA_DIR = PROJECT_ROOT + "/_data"

DATA_MNIST_DIR = DATA_DIR + "/mnist/"
DATA_FASHIONMNIST_DIR = DATA_DIR + "/fashion_mnist/"
DATA_CIFAR10_DIR = DATA_DIR + "/cifar10/"
DATA_CATSDOGS_DIR = DATA_DIR + "/cats_dogs/"

DATA_MNIST_LABELS = [
'0',
'1',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9'
]

DATA_FASHION_LABELS = [
'T-shirt/top',
'Trouser',
'Pullover',
'Dress',
'Coat',
'Sandal',
'Shirt',
'Sneaker',
'Bag',
'Ankle Boot'
]

DATA_CIFAR10_LABELS = [
'airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck'
]


#-------------------------------------------------------------------#
# WEIGHTS PATHS

WEIGHT_PATH = PROJECT_ROOT + "/_weights/"
