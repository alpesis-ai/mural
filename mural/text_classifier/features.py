import numpy as np


def features_padding(text_ints, seq_length):
    """
    Returns features of text_ints, where each text is padded with 0s or
    truncated to the input seq_length.
    """
    features = np.zeros((len(text_ints), seq_length), dtype=int)
    
    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features
