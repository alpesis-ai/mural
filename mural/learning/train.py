import settings


def train(epochs, train_loader, model_cls, loss_fn, optimizer_fn):
    for i in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            output = model_cls(images)
            loss = loss_fn(output, labels)

            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")


def train_with_steps(train_loader, model_cls, loss_fn, optimizer_fn):
    running_loss = 0
    model_cls.train()
    for images, labels in train_loader:
        images, labels = images.to(settings.DEVICE), labels.to(settings.DEVICE)
        optimizer_fn.zero_grad()
        output = model_cls.forward(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer_fn.step()
        running_loss += loss.item()
    return running_loss
