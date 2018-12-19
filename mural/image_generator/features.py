import torch

import settings
import common.data.generator
from common.visualizers.images import tensorimage_show_single, tensorimage_show_double
from image_generator.datasets import image_load, image_convert


def gram_matrix(tensor):
    """
    Calculating the Gram Matrix of a given tensor
    Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    # get the batch_size, depth, height, and width of the Tensor
    _, depth, height, width = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    return gram


def features_generate(image, model, layers):
    """
    Running an image forward through a model and getting the features for
    a set of layers.
    """
    features = {}
    x = image

    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def define_features(modelname, model, image):
    if modelname not in settings.MODELS:
        print("model name error")
        exit(1)
    
    if modelname == "VGG19_FEATURES":
        features = features_generate(image, model, common.data.generator.VGG19_FEATURE_LAYERS)
    return features


def define_style_weights(modelname):
    if modelname not in settings.MODELS:
        print("model name error")
        exit(1)
    
    if modelname == "VGG19_FEATURES":
        style_weights = common.data.generator.VGG19_STYLE_WEIGHTS
    return style_weights


def feature_style_gram(modelname, model, content, style):
    content_features = define_features(modelname, model, content)
    style_features = define_features(modelname, model, style)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    return content_features, style_grams


def content_generate(modelname, target_features, content_features):
    if modelname not in settings.MODELS:
        print("model name error")
        exit(1)
    
    if modelname == "VGG19_FEATURES":
        layer = common.data.generator.VGG19_CONTENT_LAYER

    content_loss = torch.mean((target_features[layer] - content_features[layer])**2)
    return content_loss


def style_generate(modelname, target_features, style_grams):
    style_loss = 0
    style_weights = define_style_weights(modelname)

    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, depth, height, width = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (depth * height * width)

    return style_loss


def style_transfer(target, content_features, style_grams, modelname, model, optimizer, epochs, imageloop):
    for i in range(1, epochs + 1):
        target_features = define_features(modelname, model, target)
        content_loss = content_generate(modelname, target_features, content_features)
        style_loss = style_generate(modelname, target_features, style_grams) 
        total_loss = common.data.generator.CONTENT_WEIGHT * content_loss + \
                     common.data.generator.STYLE_WEIGHT * style_loss

        # update target image
        optimizer.zero_grad()
        # total_loss.backward()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # display intermediate images and print the loss
        if  i % imageloop == 0:
            print('Total loss: ', total_loss.item())
            tensorimage_show_single(image_convert(target))

    return target
