import numpy as np
from PIL import Image
from torchvision import transforms


def image_load(image_path, max_size=400, shape=None):
    """
    Load an image in and transform it with the size <= 400 pixels in the x-y dims.
    """
    image = Image.open(image_path).convert('RGB')
    # rescale image for speeding up processing
    if max(image.size) > max_size: size = max_size
    else: size = max(image.size)

    if shape is not None: size = shape

    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = transform(image)[:3,:,:].unsqueeze(0)
    return image


def image_convert(tensor):
    """
    Display a tensor as an image.
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.0485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image
