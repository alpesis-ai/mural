import settings
from processors.images import image_load, image_convert
from visualizers.images import image_show_double


if __name__ == '__main__':
    content = image_load(settings.DATA_STYLE_TRANSFER_DIR + "octopus.jpg")
    style = image_load(settings.DATA_STYLE_TRANSFER_DIR + "hockney.jpg",
                       shape=content.shape[-2:])

    image_show_double(image_convert(content), image_convert(style))
