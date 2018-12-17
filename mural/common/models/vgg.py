from torchvision import models


def vgg19_features():
    model = models.vgg19(pretrained=True).features
    return model
