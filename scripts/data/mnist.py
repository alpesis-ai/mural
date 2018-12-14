from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.MNIST("./_data/mnist/", download=True, train=True, transform=transform)
test_data = datasets.MNIST("./_data/mnist/", download=True, train=False, transform=transform)
