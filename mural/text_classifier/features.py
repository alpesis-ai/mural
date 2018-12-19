from string import punctuation
from collections import Counter

import numpy as np


def text2words(text):
    text = text.lower()
    text_full = ''.join([c for c in text if c not in punctuation])
    text_split = text_full.split('\n')
    text_full = ' '.join(text_split)
    words = text_full.split()
    print(words[:30])
    return words, text_split


def worddict_generate(words):
    counts = Counter(words)
    vocabulary = sorted(counts, key=counts.get, reverse=True)
    vocabulary2int = {word: i for i, word in enumerate(vocabulary, 1)}
    return vocabulary2int


def text2int_generate(text_split, vocabulary2int):
    text2int = []
    for sentence in text_split:
        text2int.append([vocabulary2int[word] for word in sentence.split()]) 

    print('Unique words: ', len((vocabulary2int)))
    print('Tokenized text: \n', text2int[:1])
    return text2int


def labels_encoded(labels):
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
    return encoded_labels
 

def features_padding(text_ints, seq_length):
    """
    Returns features of text_ints, where each text is padded with 0s or
    truncated to the input seq_length.
    """
    features = np.zeros((len(text_ints), seq_length), dtype=int)
    
    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features
