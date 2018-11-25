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
