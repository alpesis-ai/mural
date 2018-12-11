import torch


def test_with_steps(test_loader, model_fn, loss_fn):
    test_loss = 0
    accuracy = 0
    
    # turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for images, labels in test_loader:
            log_probabilities = model_fn(images)
            test_loss += loss_fn(log_probabilities, labels)
            probabilities = torch.exp(log_probabilities)
            top_probability, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    return test_loss, accuracy
