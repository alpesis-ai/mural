import numpy as np

from text_classifier.learn_train import train_single
from text_classifier.learn_test import validate_with_steps


def validate_steps(epochs, train_loader, valid_loader, test_loader, model_cls, loss_fn, optimizer, batch_size, clip, evalloop):
    steps = 0
    model_cls.train()
    for epoch in range(epochs):
        hidden = model_cls.init_hidden(batch_size)
        for inputs, labels in train_loader:
            steps += 1
            train_loss = train_single(inputs, labels, hidden, model_cls, loss_fn, optimizer, clip)
            if steps % evalloop == 0:
                valid_losses = validate_with_steps(valid_loader, model_cls, loss_fn, batch_size)
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Step: {}...".format(steps),
                      "Train Loss: {:.6f}...".format(train_loss.item()),
                      "Validation Loss: {:.6f}".format(np.mean(valid_losses)))
