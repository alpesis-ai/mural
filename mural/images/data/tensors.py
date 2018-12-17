import torch


def gram_matrix(tensor):
    """
    Calculating the Gram Matrix of a given tensor
    Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    # get the batch_size, depth, height, and width of the Tensor
    _, depth, height, width = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    return gram
