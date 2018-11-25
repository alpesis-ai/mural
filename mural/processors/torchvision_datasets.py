import torch
from torchvision import datasets, transforms

import settings


def data_loader(dataset):
    # normalize data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == "MNIST":
        train_data = datasets.FashionMNIST(settings.DATA_MNIST_DIR, download=False, train=True, transform=transform)
        test_data = datasets.FashionMNIST(settings.DATA_MNIST_DIR, download=False, train=False, transform=transform)

    elif dataset == "FASHIONMNIST":
        train_data = datasets.FashionMNIST(settings.DATA_FASHIONMNIST_DIR, download=False, train=True, transform=transform)
        test_data = datasets.FashionMNIST(settings.DATA_FASHIONMNIST_DIR, download=False, train=False, transform=transform)
 
    else:
        print("Dataset input error!")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    return train_loader, test_loader
