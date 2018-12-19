import settings
from common.models.classifiers import Classifier, ClassifierWithDropout
from common.models.mlp import MLP
from common.models.cnn import CNN
from common.models.densenet import densenet121_trans
from common.models.vgg import vgg19_features
from common.models.charrnn import CharRNN


def define_model(name, values, num_hidden, num_layers):
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
    elif (name == "CHARRNN"):
        model = CharRNN(values, num_hidden, num_layers)

    print("{} Model Structure:".format(name))
    print(model)
    return model
