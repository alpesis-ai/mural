import torch
from torch import nn
import numpy as np

import settings.common
from models.charrnn import CharRNN
from processors.texts import tokenize, onehot_encode, get_batches

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            x = onehot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            inputs, targets = inputs.to(settings.common.DEVICE), targets.to(settings.common.DEVICE)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()
            output, h = net(inputs, h)
            
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = onehot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    inputs, targets = inputs.to(settings.common.DEVICE), targets.to(settings.common.DEVICE)

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length))
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

if __name__ == '__main__':
    with open(settings.common.DATA_CHARRNN_DIR + 'dummy.txt', 'r') as f:
        text = f.read()
    encoded = tokenize(text)

    batch_size = 8
    seq_length = 50
    batches = get_batches(encoded, batch_size, seq_length)
    x, y = next(batches)
    print('x\n', x[:10, :10])
    print('\ny\n', y[:10, :10])

    # define and print the net
    n_hidden=512
    n_layers=2
    chars = tuple(set(text))
    net = CharRNN(chars, n_hidden, n_layers)
    print(net)

    batch_size = 18
    seq_length = 10
    n_epochs = 2
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=1)
