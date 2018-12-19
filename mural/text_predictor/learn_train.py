import torch
from torch import nn

import settings
from text_predictor.text_processing import onehot_encode, get_batches


def train_single(inputs, targets, hidden, batch_size, seq_length, n_chars, model_cls, loss_fn, optimizer, clip):
    model_cls.train()
    inputs = onehot_encode(inputs, n_chars)
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    inputs = inputs.to(settings.DEVICE)
    targets = targets.to(settings.DEVICE)
        
    # creating new variables for hidden state
    # otherwise, we'd backprop through the entire training history
    hidden = tuple([each.data for each in hidden])

    model_cls.zero_grad()
    outputs, hidden = model_cls(inputs, hidden)
    loss = loss_fn(outputs, targets.view(batch_size * seq_length))
    loss.backward()
    # clip_grad_norm: preventing the exploding gradient problem in RNNs/ LSTMs
    nn.utils.clip_grad_norm_(model_cls.parameters(), clip)
    optimizer.step()

    return loss
