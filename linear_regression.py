import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def linear_regression(x, y):
    # Average for X & Y
    mean_x, mean_y = np.mean(x), np.mean(y)

    total_xy = np.sum((x-mean_x)*(y-mean_y))

    total_xx = np.sum(x*(x-mean_x))

    slope = total_xy / total_xx

    b = mean_y - (slope*mean_x)

    return (slope, b)


def plot_regression(x, y, b):
    plt.scatter(x, y, color="blue", marker="o", s=30)

    y_pred = b[0]*x + b[1]

    plt.plot(x, y_pred, lw=4,  c='green')
    plt.xlabel('X-Independent_variable SAT Exam score')
    plt.ylabel('Y-Dependent_variable GPA Grade point average')

    plt.show()


def main():
    # Dataset
    data = pd.read_csv('sample_linear_regression.csv')
    data.describe()
    x = data['SAT']
    y = data['GPA']

    # Plot Dataset
    plt.scatter(x, y)
    plt.xlabel('SAT')
    plt.ylabel('GPA')
    plt.show()

    b = linear_regression(x, y)
    print("Values from linear regression slope = {} and y = {}"
          .format(b[0], b[1]))

    # Predictions
    sample_sat = x[random.randint(0, 84)]
    gpa_prediction = b[0]*sample_sat + b[1]
    print("Prediction for a SAT score of {} it will receive a GPA score of {}"
          .format(sample_sat, gpa_prediction))

    plot_regression(x, y, b)


if __name__ == '__main__':
    main()
