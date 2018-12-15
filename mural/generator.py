import argparse

import settings.common
import settings.generator
from learning.optimizers import define_optimizer_generator
from learning.features import feature_style_gram, style_transfer
from learning.models import define_model
from processors.images import image_load, image_convert
from visualizers.images import tensorimage_show_single, tensorimage_show_double


def set_params():
    parser = argparse.ArgumentParser(description='Mural Generator Parameters')

    parser.add_argument('--model',
                        type=str,
                        help="Model: [VGG19_FEATURES]")

    parser.add_argument('--optimizer',
                        type=str,
                        help="Optimizer: [ADAM]")

    parser.add_argument('--rate',
                        type=float,
                        help="Learning Rate: e.g. 0.03")

    parser.add_argument('--epochs',
                        type=int,
                        help="epochs")

    parser.add_argument('--imageloop',
                        type=int,
                        help="imageloop: display image result per loop, e.g. 400")

    return parser.parse_args()


if __name__ == '__main__':

    args = set_params()

    content = image_load(settings.common.DATA_STYLE_TRANSFER_DIR + "octopus.jpg")
    style = image_load(settings.common.DATA_STYLE_TRANSFER_DIR + "hockney.jpg", shape=content.shape[-2:])
    tensorimage_show_double(image_convert(content), image_convert(style))
    content.to(settings.common.DEVICE)
    style.to(settings.common.DEVICE)

    model = define_model(args.model)
    model.to(settings.common.DEVICE)
    content_features, style_grams = feature_style_gram(args.model, model, content, style)

    target = content.clone().requires_grad_(True)
    target.to(settings.common.DEVICE)
    optimizer = define_optimizer_generator(args.optimizer, args.rate, [target])
    target = style_transfer(target, content_features, style_grams,
                            args.model, model, optimizer, args.epochs, args.imageloop)
    tensorimage_show_double(image_convert(content), image_convert(target))
