import settings
from images.models.classifiers import Classifier, ClassifierWithDropout
from images.models.mlp import MLP
from images.models.cnn import CNN
from images.models.densenet import densenet121_trans
from images.models.vgg import vgg19_features


def define_model(name):
    if name not in settings.MODELS:
        print("model name error")
        exit(1)

    if (name == "CLASSIFIER"):
        model = Classifier()
    elif (name == "CLASSIFIER_DROPOUT"):
        model = ClassifierWithDropout()
    elif (name == "MLP"):
        model = MLP()
    elif (name == "CNN"):
        model = CNN()
    elif (name == "DENSENET121_TRANS"):
        model = densenet121_trans()
    elif (name == "VGG19_FEATURES"):
        model = vgg19_features()

    print("{} Model Structure:".format(name))
    print(model)
    return model
