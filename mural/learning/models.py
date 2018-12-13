from models.classifiers import Classifier, ClassifierWithDropout
from models.mlp import MLP
from models.densenet import densenet121_trans


def define_model(name):
    if (name == "CLASSIFIER"):
        model = Classifier()
        return model

    elif (name == "CLASSIFIER_DROPOUT"):
        model = ClassifierWithDropout()
        return model

    elif (name == "MLP"):
        model = MLP()
        return model

    elif (name == "DENSENET121_TRANS"):
        model = densenet121_trans()
        return model

    else:
        print("model name error")
        exit(1)
