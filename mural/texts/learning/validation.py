import numpy as np

from texts.learning.train import train_single
from texts.learning.test import valid_with_steps
from texts.data.texts import get_batches


def validate_steps(epochs, train_data, valid_data, model_cls, loss_fn, optimizer, batch_size, seq_length, clip, imageloop):
    steps = 0
    n_chars = len(model_cls.chars)

    model_cls.train()
    for epoch in range(epochs):
        hidden = model_cls.init_hidden(batch_size)

        for x, y in get_batches(train_data, batch_size, seq_length):
            steps += 1
            train_loss = train_single(x, y, hidden, batch_size, seq_length, n_chars, model_cls, loss_fn, optimizer, clip)

            if steps % imageloop == 0:
                valid_losses = valid_with_steps(valid_data, batch_size, seq_length, n_chars, model_cls, loss_fn)
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Step: {}...".format(steps),
                      "Train Loss: {:.4f}...".format(train_loss.item()),
                      "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
