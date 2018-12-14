from torchvision import datasets, transforms

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = datasets.CIFAR10("./data/cifar10/", download=True, train=True, transform=transform)
test_data = datasets.CIFAR10("./data/cifar10/", download=True, train=False, transform=transform)
