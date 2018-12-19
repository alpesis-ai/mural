import argparse

import settings
from common.managers.models import define_model_texts
from common.managers.losses import define_loss
from common.managers.optimizers import define_optimizer_classifier
from texts.learning.validation import validate_steps
from texts.data.texts import tokenize, onehot_encode, get_batches


def set_params():
    parser = argparse.ArgumentParser(description='Mural Classifier Parameters')

    parser.add_argument('--loss',
                        type=str,
                        required=True,
                        help="Loss: [NLL, CROSSENTROPY]")
    
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help="Model: [CHARRNN]")

    parser.add_argument('--optimizer',
                        type=str,
                        help="Optimizer: [ADAM, SGD, ADAM_TRANS, SGD_TRANS]")

    parser.add_argument('--rate',
                        type=float,
                        help="Learning Rate: [e.g. 0.001]")

    parser.add_argument('--epochs',
                        type=int,
                        help="epochs (train only)")

    parser.add_argument('--batchsize',
                        type=int,
                        help="batch size")

    parser.add_argument('--seqlength',
                        type=int,
                        help="sequence length")

    parser.add_argument('--clip',
                        type=int,
                        help="clip")

    parser.add_argument('--imageloop',
                        type=int,
                        help="imageloop (train only)")

    return parser.parse_args()


if __name__ == '__main__':
    
    args = set_params()

    with open(settings.DATA_CHARRNN_DIR + 'dummy.txt', 'r') as f:
        text = f.read()
    encoded = tokenize(text)
    valid_idx = int(len(encoded)*(1-settings.DATA_VALID_SIZE))
    train_data, valid_data = encoded[:valid_idx], encoded[valid_idx:]

    n_hidden=512
    n_layers=2
    chars = tuple(set(text))
    model = define_model_texts(args.model, chars, n_hidden, n_layers)

    optimizer = define_optimizer_classifier(args.optimizer, args.rate, model)
    criterion = define_loss(args.loss)
    validate_steps(args.epochs, train_data, valid_data, model, criterion, optimizer, args.batchsize, args.seqlength, args.clip, args.imageloop)
