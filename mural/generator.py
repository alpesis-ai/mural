import torch
from torch import optim

import settings.common
import settings.generator
from models.vgg import vgg19_features
from processors.images import image_load, image_convert
from processors.features import features_generate
from processors.tensors import gram_matrix
from visualizers.images import tensorimage_show_single, tensorimage_show_double


def style_transfer(target, content_features, style_grams, model):
    optimizer = optim.Adam([target], lr=0.003)
    for i in range(1, settings.generator.STEPS + 1):
        # get the features from your target image
        target_features = features_generate(target, model, settings.generator.VGG19_FEATURE_LAYERS)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
       
        style_loss = 0
        style_weights = settings.generator.VGG19_STYLE_WEIGHTS
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, depth, height, width = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (depth * height * width)
        # calculate the total loss
        total_loss = settings.generator.CONTENT_WEIGHT * content_loss + settings.generator.STYLE_WEIGHT * style_loss
        # update target image
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step() 
 
        # display intermediate images and print the loss
        if  i % settings.generator.SHOW_EVERY == 0:
            print('Total loss: ', total_loss.item())
            tensorimage_show_single(image_convert(target))

    return target


if __name__ == '__main__':
    content = image_load(settings.common.DATA_STYLE_TRANSFER_DIR + "octopus.jpg")
    style = image_load(settings.common.DATA_STYLE_TRANSFER_DIR + "hockney.jpg", shape=content.shape[-2:])
    tensorimage_show_double(image_convert(content), image_convert(style))

    model = vgg19_features()
    content_features = features_generate(content, model, settings.generator.VGG19_FEATURE_LAYERS)
    style_features = features_generate(style, model, settings.generator.VGG19_FEATURE_LAYERS)
    # calculate the gram matrices for each layer of style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)
    target = style_transfer(target, content_features, style_grams, model)
    tensorimage_show_double(image_convert(content), image_convert(target))
