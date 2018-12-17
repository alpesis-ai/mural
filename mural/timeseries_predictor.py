import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from timeseries.models.rnn import RNN
from timeseries.data.timeseries import timesteps_generate, timeseries_generate
from timeseries.visualizers.data import scatter_plot

def train(rnn, n_steps, print_every):
    
    # initialize the hidden state
    hidden = None      
    
    for batch_i, step in enumerate(range(n_steps)):
        time_steps = timesteps_generate(seq_length, step)
        data = timeseries_generate(time_steps)
        data.resize((seq_length + 1, 1)) # input_size=1
        x = data[:-1]
        y = data[1:]
        
        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i%print_every == 0:        
            print('Loss: ', loss.item())
            scatter_plot(time_steps[1:], x, prediction.data.numpy().flatten(), "input, x", "target, y")
    
    return rnn

if __name__ == '__main__':
    seq_length = 20

    input_size = 1
    output_size = 1
    hidden_dim = 32
    n_layers = 1
    rnn = RNN(input_size, output_size, hidden_dim, n_layers)
    print(rnn)

    # MSE loss and Adam optimizer with a learning rate of 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

    # train the rnn and monitor results
    n_steps = 75
    print_every = 15
    trained_rnn = train(rnn, n_steps, print_every) 
