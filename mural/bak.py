import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
    train_data = datasets.MNIST('../data/mnist/', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)
    # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

    model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
    
            optimizer.zero_grad()
        
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(train_loader)}")

    #images, labels = next(iter(train_loader))

    #img = images[0].view(1, 784)
    # Turn off gradients to speed up this part
    #with torch.no_grad():
    #    logits = model.forward(img)

    # Output of the network are logits, need to take softmax for probabilities
    #ps = F.softmax(logits, dim=1)
    #helper.view_classify(img.view(1, 28, 28), ps)

if __name__ == '__main__':
    main()
    
