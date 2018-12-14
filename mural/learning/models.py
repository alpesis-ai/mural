import settings
from models.classifiers import Classifier, ClassifierWithDropout
from models.mlp import MLP
from models.cnn import CNN
from models.densenet import densenet121_trans


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

    print(name)
    print(model)
    return model
