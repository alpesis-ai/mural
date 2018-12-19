import settings
from image_classifier.datasets import generate_loader, load_mnist, load_fashionmnist, load_cifar10, load_catsdogs
from text_predictor.datasets import generate_data, load_dummy, load_anna

def define_dataset(name):
    if name not in settings.DATASETS:
        print("Dataset input error.")
        exit(1)

    if (name == "MNIST"):
        train_data, test_data = load_mnist()
    elif (name == "FASHIONMNIST"):
        train_data, test_data = load_fashionmnist()
    elif (name == "CIFAR10"):
        train_data, test_data = load_cifar10()
    elif (name == "CATSDOGS"):
        train_data, test_data = load_catsdogs()

    train_loader, valid_loader, test_loader = generate_loader(train_data, test_data)
    return train_loader, valid_loader, test_loader


def define_dataset_texts(name):
    if name not in settings.DATASETS:
        print("Dataset input error.")
        exit(1)

    if (name == "BOOK_DUMMY"):
        train_raw = load_dummy()
    elif (name == "BOOK_ANNA"):
        train_raw = load_anna()

    train_data, valid_data = generate_data(train_raw)
    return train_raw, train_data, valid_data
