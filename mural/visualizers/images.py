import numpy as np
import matplotlib.pyplot as plt


def image_show(image, ax=None, title=None, normalize=True):
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


def image_predict(image, probabilities, labels):
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
