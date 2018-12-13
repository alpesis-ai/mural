import torch
from torchvision import datasets, transforms

import settings


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.FashionMNIST(settings.DATA_MNIST_DIR, download=False, train=True, transform=transform)
    test_data = datasets.FashionMNIST(settings.DATA_MNIST_DIR, download=False, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=settings.DATA_BATCH_SIZE,
                                               num_workers=settings.DATA_NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=settings.DATA_BATCH_SIZE,
                                              num_workers=settings.DATA_NUM_WORKERS)
    return train_loader, test_loader


def load_fashionmnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.FashionMNIST(settings.DATA_FASHIONMNIST_DIR, download=False, train=True, transform=transform)
    test_data = datasets.FashionMNIST(settings.DATA_FASHIONMNIST_DIR, download=False, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=settings.DATA_BATCH_SIZE,
                                               num_workers=settings.DATA_NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=settings.DATA_BATCH_SIZE,
                                              num_workers=settings.DATA_NUM_WORKERS)
    return train_loader, test_loader
   

def load_catsdogs():
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(settings.DATA_CATSDOGS_DIR + '/train', transform=train_transform)
    test_data = datasets.ImageFolder(settings.DATA_CATSDOGS_DIR + '/test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=settings.DATA_BATCH_SIZE,
                                               num_workers=settings.DATA_NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=settings.DATA_BATCH_SIZE,
                                              num_workers=settings.DATA_NUM_WORKERS)
    return train_loader, test_loader


def define_dataset(name):
    if (name == "MNIST"):
        train_loader, test_loader = load_mnist()
        return train_loader, test_loader

    elif (name == "FASHIONMNIST"):
        train_loader, test_loader = load_fashionmnist()
        return train_loader, test_loader

    elif (name == "CATSDOGS"):
        train_loader, test_loader = load_catsdogs()
        return train_loader, test_loader

    else:
        print("Dataset input error.")
        exit(1) 
