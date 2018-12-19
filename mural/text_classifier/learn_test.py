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
