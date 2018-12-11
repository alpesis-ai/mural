from torchvision import models
from models.classifiers import Classifier, ClassifierWithDropout


def define_model(name):
    if (name == "CLASSIFIER"):
        model = Classifier()
        return model

    elif (name == "CLASSIFIER_DROPOUT"):
        model = ClassifierWithDropout()
        return model

    elif (name == "DENSENET121"):
        model = models.densenet121(pretrained=True)
        return model

    else:
        print("model name error")
        exit(1)
