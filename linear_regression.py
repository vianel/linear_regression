import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x,y):
    n = np.size(x)
    # Average for X & Y
    mean_x, mean_y = np.mean(x), np.mean(y)

    total_xy = np.sum((x-mean_x)*(y-mean_y))

    total_xx = np.sum(x*(x-mean_x))

    slope = total_xy / total_xx

    b = mean_y - (slope*mean_x)

    return (slope, b)

def plot_regression(x, y, b):
    plt.scatter(x, y, color = "g", marker = "o", s = 30)

    y_pred = b[1]*x + b[0]

    plt.plot(x, y_pred, color = "b")
    plt.xlabel('X-Independent_variable')
    plt.ylabel('Y-Dependent_variable')

    plt.show()


def main():
    #Dataset
    x = np.array([1,2,3,4,5])
    y = np.array([2,3,5,6,5])

    b = linear_regression(x, y)
    print("Values from linear regression slope = {} and y = {}".format(b[0], b[1]))

    plot_regression(x, y, b)

if __name__ == '__main__':
    main()


