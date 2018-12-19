import matplotlib.pyplot as plt


def scatter_plot(distance, x, y, label_x, label_y):
    plt.figure(figsize=(8, 5))
    plt.plot(distance, x, 'r.', label=label_x)
    plt.plot(distance, y, 'b.', label=label_y)
    plt.legend(loc='best')
    plt.show()
