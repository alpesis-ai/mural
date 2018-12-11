import matplotlib.pyplot as plt


def loss_plot(losses, label):
    plt.plot(losses, label=label)
    plt.legend(frameon=False)
    plt.show()
