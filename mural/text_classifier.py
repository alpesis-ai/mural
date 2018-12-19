from string import punctuation
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import settings
from common.data.text import text_load
from common.models.sentimentrnn import SentimentRNN
from common.managers.losses import define_loss
from common.managers.optimizers import define_optimizer_classifier
from text_classifier.dataset_processing import remove_outlier
from text_classifier.datasets import get_loader
from text_classifier.features import text2words, worddict_generate, text2int_generate, labels_encoded, features_padding
from text_classifier.learn_validation import validate_steps


if __name__ == '__main__':
    reviews = text_load(settings.DATA_SENTIMENT_DIR + 'reviews.txt')
    labels = text_load(settings.DATA_SENTIMENT_DIR + 'labels.txt')
    
    words, text_split = text2words(reviews)
    vocabulary2int = worddict_generate(words)
    text2int = text2int_generate(text_split, vocabulary2int)
    encoded_labels = labels_encoded(labels)
    text2int, encoded_labels = remove_outlier(text2int, encoded_labels)

    seq_length = 200
    train_loader, valid_loader, test_loader = get_loader(text2int, encoded_labels, seq_length)
    batch_size = 20
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocabulary2int)+1 # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)

    lr=0.001
    criterion = define_loss("BCE")
    optimizer = define_optimizer_classifier("ADAM", lr, net)
    epochs = 2
    clip=5
    evalloop = 1
    validate_steps(epochs, train_loader, valid_loader, net, criterion, optimizer, batch_size, clip, evalloop)
