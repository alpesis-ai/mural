import torch
import numpy as np

import settings


def validate_with_steps(valid_loader, model_cls, loss_fn, batch_size):
    valid_losses = []
    valid_hidden = model_cls.init_hidden(batch_size)
    
    model_cls.eval()
    for inputs, labels in valid_loader:
        valid_hidden = tuple([each.data for each in valid_hidden])
        inputs = inputs.to(settings.DEVICE)
        labels = labels.to(settings.DEVICE)

        output, valid_hidden = model_cls(inputs, valid_hidden)
        valid_loss = loss_fn(output.squeeze(), labels.float())
        valid_losses.append(valid_loss.item())
    return valid_losses


def test_with_multi(test_loader, model_cls, loss_fn, batch_size):
    test_losses = []
    num_correct = 0

    model_cls.eval()
    hidden = model_cls.init_hidden(batch_size)

    for inputs, labels in test_loader:
        hidden = tuple([each.data for each in hidden])
        inputs = inputs.to(settings.DEVICE)
        labels = labels.to(settings.DEVICE)

        output, hidden = model_cls(inputs, hidden)
        test_loss = loss_fn(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        prediction = torch.round(output.squeeze())
        # compare predicitions to true label
        correct_tensor = prediction.eq(labels.float().view_as(prediction))
        correct = np.squeeze(correct_tensor.to(settings.DEVICE).numpy())
        num_correct += np.sum(correct)
   
    print("Test Loss: {:.3f}".format(np.mean(test_losses)))
    test_accuracy = num_correct / len(test_loader.dataset)
    print("Test Accuracy: {:.3f}".format(test_accuracy))

    return test_losses
