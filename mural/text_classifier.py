from string import punctuation
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import settings
from texts.data.datasets import load_text
from texts.data.features import features_padding
from common.models.sentimentrnn import SentimentRNN


if __name__ == '__main__':
    reviews = load_text(settings.DATA_SENTIMENT_DIR + 'reviews.txt')
    labels = load_text(settings.DATA_SENTIMENT_DIR + 'labels.txt')
    print(reviews[:1000])
    print()
    print(labels[:20])

    # remove punctuation
    reviews = reviews.lower() # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])
    # split sentences
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()
    print(words[:30])

    ## Build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    ## use the dict to tokenize each review in reviews_split
    ## store the tokenized reviews in reviews_ints
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])
    # stats about vocabulary
    print('Unique words: ', len((vocab_to_int)))
    # print tokens in first review
    print('Tokenized review: \n', reviews_ints[:1])

    # 1=positive, 0=negative label conversion
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

    # outlier review stats
    review_lens = Counter([len(x) for x in reviews_ints])
    print("Zero-length reviews: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))
    print('Number of reviews before removing outliers: ', len(reviews_ints))
    ## remove any reviews/labels with zero length from the reviews_ints list.
    # get indices of any reviews with length 0
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    # remove 0-length reviews and their labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
    print('Number of reviews after removing outliers: ', len(reviews_ints))

    seq_length = 200
    features = features_padding(reviews_ints, seq_length=seq_length)
    ## test statements - do not change - ##
    assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
    assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
    # print first 10 values of the first 30 batches 
    print(features[:30,:10])

    split_frac = 0.8
    ## split data into training, validation, and test data (features and labels, x and y)
    split_idx = int(len(features)*0.8)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x)*0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape), 
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    batch_size = 50
    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)

    # loss and optimization functions
    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
