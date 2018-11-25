import os.path

#-------------------------------------------------------------------#
# ROOT SETTINGS

# root path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

#-------------------------------------------------------------------#
# DATA PATHS

DATA_DIR = PROJECT_ROOT + "/data"

# MNIST
DATA_DIR_MNIST = DATA_DIR + "mnist"

# Fashion
DATA_DIR_FASHION = DATA_DIR + "/fashion_mnist"

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
