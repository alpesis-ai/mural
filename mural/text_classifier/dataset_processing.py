from collections import Counter

import numpy as np


def remove_outlier(text2int, encoded_labels):
    text2int_length = Counter([len(x) for x in text2int])
    print("Zero-length text: {}".format(text2int_length[0]))
    print("Maximum text length: {}".format(max(text2int_length)))
    print('Number of text before removing outliers: ', len(text2int))

    # remove any reviews/labels with zero length from the reviews_ints list
    # get indices of any text with length 0
    non_zero_idx = [i for i, text in enumerate(text2int) if len(text) != 0]
    # remove 0-length reviews and their labels
    text2int = [text2int[i] for i in non_zero_idx]
    print('Number of text after removing outliers: ', len(text2int))
    encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx])
    return text2int, encoded_labels
