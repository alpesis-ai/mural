import numpy as np
import matplotlib.pyplot as plt

import settings
from processors.data_selection import select_data_single, select_data_multi


def image_show_single(image, ax=None, title=None, normalize=True):
    """
    Showing image with tensor format.
    """
    if ax is None:
        figure, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show(ax)
    return ax


def image_show_multi(images, labels):
    images = images.numpy()
    figure = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = figure.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(str(labels[idx].item()))
    plt.show()


def image_show_detail(images):
    images = images.numpy()
    image = np.squeeze(images[1])

    figure = plt.figure(figsize = (12,12)) 
    ax = figure.add_subplot(111)
    ax.imshow(image, cmap='gray')
    width, height = image.shape
    thresh = image.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(image[x][y],2) if image[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if image[x][y] < thresh else 'black')
    plt.show()


def image_predict_single(image, probabilities, labels):
    """
    Viewing a predicted image and its predicted classes.
    """
    probs = probabilities.data.numpy().squeeze()

    figure, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))

    ax2.set_yticklabels(labels, size="small");
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)
    
    plt.tight_layout()

    plt.show((ax1, ax2))


def image_predict_multi(images, predicted_labels, labels):
    images = images.numpy()

    figure = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = figure.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap="gray")
        ax.set_title("{} ({})".format(
                          str(predicted_labels[idx].item()),
                          str(labels[idx].item())),
                      color=("green" if predicted_labels[idx] == labels[idx] else "red"))
    plt.show()


def preshow_images(data_loader):
    if (settings.IMAGE_EXPLORE == 1):
        image, label = select_data_single(data_loader)
        print(image.shape, label.shape)
        image_show_single(image[0, :])
    elif (settings.IMAGE_EXPLORE == 2):
        images, labels = select_data_multi(data_loader)
        image_show_multi(images, labels)
    elif (settings.IMAGE_EXPLORE == 3):
        images, labels = select_data_multi(data_loader)
        image_show_detail(images) 
