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
