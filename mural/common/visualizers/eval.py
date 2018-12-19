import matplotlib.pyplot as plt


def loss_compare(losses1, losses2, label1, label2):
    plt.plot(losses1, label=label1)
    plt.plot(losses2, label=label2)
    plt.legend(frameon=False)
    plt.show()
