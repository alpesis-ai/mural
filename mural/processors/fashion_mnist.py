import torch
from torchvision import datasets, transforms


def data_loader(data_path):
    # normalize data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    return train_loader, test_loader
