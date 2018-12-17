import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import settings
from images.data.data_selection import split_dataset


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.FashionMNIST(settings.DATA_MNIST_DIR,
                                       download=False,
                                       train=True,
                                       transform=transform)
    test_data = datasets.FashionMNIST(settings.DATA_MNIST_DIR,
                                      download=False,
                                      train=False,
                                      transform=transform)
    return train_data, test_data


def load_fashionmnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.FashionMNIST(settings.DATA_FASHIONMNIST_DIR,
                                       download=False,
                                       train=True,
                                       transform=transform)
    test_data = datasets.FashionMNIST(settings.DATA_FASHIONMNIST_DIR,
                                      download=False,
                                      train=False,
                                      transform=transform)
    return train_data, test_data
   

def load_cifar10():
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), # randomly flip and rotate
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(settings.DATA_CIFAR10_DIR,
                                  train=True,
                                  download=False,
                                  transform=transform)
    test_data = datasets.CIFAR10(settings.DATA_CIFAR10_DIR,
                                 train=False,
                                 download=False,
                                 transform=transform) 
    return train_data, test_data


def load_catsdogs():
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(settings.DATA_CATSDOGS_DIR + '/train',
                                      transform=train_transform)
    test_data = datasets.ImageFolder(settings.DATA_CATSDOGS_DIR + '/test',
                                     transform=test_transform)
    return train_data, test_data


def generate_loader(train_data, test_data):
    num_train = len(train_data)
    train_idx, valid_idx = split_dataset(num_train)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=settings.DATA_BATCH_SIZE,
                                               sampler=train_sampler,
                                               num_workers=settings.DATA_NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=settings.DATA_BATCH_SIZE, 
                                               sampler=valid_sampler,
                                               num_workers=settings.DATA_NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=settings.DATA_BATCH_SIZE, 
                                              num_workers=settings.DATA_NUM_WORKERS)

    return train_loader, valid_loader, test_loader
    

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
