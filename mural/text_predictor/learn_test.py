import torch
import torch.nn.functional as F
import numpy as np

import settings
from text_predictor.text_processing import onehot_encode, get_batches


def valid_with_steps(valid_data, batch_size, seq_length, n_chars, model_cls, loss_fn):
    valid_losses = []

    valid_hidden = model_cls.init_hidden(batch_size)
    model_cls.eval()
    for x, y in get_batches(valid_data, batch_size, seq_length):
        x = onehot_encode(x, n_chars)
        inputs = torch.from_numpy(x).to(settings.DEVICE)
        targets = torch.from_numpy(y).to(settings.DEVICE)

        # creating new variables for hidden state
        # otherwise,, we'd backprop through the entire training history
        valid_hiddden = tuple([each.data for each in valid_hidden])
        outputs, valid_hidden = model_cls(inputs, valid_hidden)
        valid_loss = loss_fn(outputs, targets.view(batch_size * seq_length))
        valid_losses.append(valid_loss.item())
    return valid_losses


def test_single(test_data, model_cls, hidden=None, topk=None):
    """
    Given a character, predict the next character.
    Returns the predicted character and the hidden state.
    """
    inputs = np.array([[model_cls.char2int[test_data]]])
    inputs = onehot_encode(inputs, len(model_cls.chars))
    inputs = torch.from_numpy(inputs).to(settings.DEVICE)

    hidden = tuple([each.data for each in hidden])
    outputs, hidden = model_cls(inputs, hidden)
        
    probabilities = F.softmax(outputs, dim=1).data
    probabilities.to(settings.DEVICE)

    if topk is None:
        top_outs = np.arange(len(model_cls.chars))
    else:
        top_probs, top_outs = probabilities.topk(topk)
        top_outs = top_outs.numpy().squeeze()
       
    # select the likely next character with some element of randomness
    probs = top_probs.numpy().squeeze()
    outs = np.random.choice(top_outs, p=probs/probs.sum()) 

    return model_cls.int2char[outs], hidden
