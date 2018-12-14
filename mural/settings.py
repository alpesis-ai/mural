import os.path

import torch

#-------------------------------------------------------------------#
# PERFORMANCES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for data processor
DATA_NUM_WORKERS = 0
DATA_BATCH_SIZE = 20

#-------------------------------------------------------------------#
# Training

LEARNING_RATE = 0.01

#-------------------------------------------------------------------#
# VISUALIZATION

# image shown:
# - 0: not shown
# - 1: show a single image
# - 2: show multi images
# - 3: show image detail
IMAGE_EXPLORE = 0

#-------------------------------------------------------------------#
# ROOT SETTINGS

# root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#-------------------------------------------------------------------#
# DATA PATHS

DATA_DIR = PROJECT_ROOT + "/data"

DATA_MNIST_DIR = DATA_DIR + "/mnist/"
DATA_FASHIONMNIST_DIR = DATA_DIR + "/fashion_mnist/"
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

#-------------------------------------------------------------------#
# WEIGHTS PATHS

WEIGHT_PATH = PROJECT_ROOT + "/_weights/"
