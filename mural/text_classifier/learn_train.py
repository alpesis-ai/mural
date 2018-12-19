from torch import nn

import settings


def train_single(inputs, labels, hidden, model_cls, loss_fn, optimizer, clip):
    model_cls.train()
    inputs = inputs.to(settings.DEVICE)
    labels = labels.to(settings.DEVICE)

    # creating new variables for hidden state
    # otherwise, backprop through entire training history
    hidden = tuple([each.data for each in hidden])

    model_cls.zero_grad()
    output, hidden = model_cls(inputs, hidden)
    loss = loss_fn(output.squeeze(), labels.float())
    loss.backward()
    nn.utils.clip_grad_norm_(model_cls.parameters(), clip)
    optimizer.step()
    return loss
