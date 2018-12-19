import torch
from torch import nn
import numpy as np

import settings
from common.models.charrnn import CharRNN
from texts.learning.train import train_single
from texts.learning.test import valid_with_steps
from texts.data.texts import tokenize, onehot_encode, get_batches

def train(train_data, valid_data, model, opt, criterion, epochs=10, batch_size=10, seq_length=50, clip=5, print_every=10):
    steps = 0
    n_chars = len(model.chars)

    model.train()
    for e in range(epochs):
        h = model.init_hidden(batch_size)
        
        for x, y in get_batches(train_data, batch_size, seq_length):
            steps += 1
            loss = train_single(x, y, h, batch_size, seq_length, n_chars, model, criterion, opt, clip)
            
            if steps % print_every == 0:
                val_losses = valid_with_steps(valid_data, batch_size, seq_length, n_chars, model, criterion)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(steps),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


if __name__ == '__main__':
    with open(settings.DATA_CHARRNN_DIR + 'dummy.txt', 'r') as f:
        text = f.read()
    encoded = tokenize(text)
    valid_idx = int(len(encoded)*(1-settings.DATA_VALID_SIZE))
    train_data, valid_data = encoded[:valid_idx], encoded[valid_idx:]

    n_hidden=512
    n_layers=2
    chars = tuple(set(text))
    model = CharRNN(chars, n_hidden, n_layers)
    print(model)

    batch_size = 18
    seq_length = 10
    n_epochs = 2
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(train_data, valid_data, model, optimizer, criterion, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, print_every=1)
