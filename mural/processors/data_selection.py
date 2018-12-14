import settings

def select_data_single(data_loader):
    image, label = next(iter(data_loader))
    return image, label


def select_data_multi(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    return images, labels
