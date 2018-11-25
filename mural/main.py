import torch
from torch import nn, optim
from torchvision import datasets, transforms

import settings
from train import train
from models.perceptrons import Perceptrons
from processors.fashion_mnist import data_loader
from visualizers.images import image_show, image_predict
import helper

if __name__ == '__main__':

    train_loader, test_loader = data_loader(settings.DATA_DIR + '/fashion_mnist/')
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    image_show(image[0, :]) 

    epochs = 2
    model = Perceptrons()
    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train(epochs, train_loader, model, loss, optimizer) 

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    img = images[1]

    # calculate the class probabilities (softmax) for img
    probabilities = torch.exp(model(img))
    image_predict(img, probabilities, version="Fashion")
