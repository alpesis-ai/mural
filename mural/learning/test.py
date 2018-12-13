import torch
import settings


def test(image, model_cls):
    model_cls.eval()

    with torch.no_grad():
        output = model_cls.forward(image)
    probabilities = torch.exp(output)
    return probabilities


def test_with_steps(test_loader, model_cls, loss_fn):
    test_loss = 0
    accuracy = 0
    
    # turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model_cls.eval()
        for images, labels in test_loader:
            images, labels = images.to(settings.DEVICE), labels.to(settings.DEVICE)
            log_probabilities = model_cls.forward(images)
            test_loss += loss_fn(log_probabilities, labels)
            probabilities = torch.exp(log_probabilities)
            top_probability, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    return test_loss, accuracy
