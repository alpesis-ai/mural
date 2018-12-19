import torch

import settings
from texts.data.texts import onehot_encode, get_batches


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
