import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def graph(x, y, model):
    plt.scatter(x, y, color='b')
    plt.plot(x, model.predict(x), color='black')
    plt.title('Salary vs Experience')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.show()


def main():
    dataset = pd.read_csv('salary.csv')
    print(dataset.head(5))

    print(dataset.shape)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    print(x)
    print(x.shape)
    print('-----')
    print(y)
    print(y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    graph(X_train, Y_train, regressor)

    graph(X_test, Y_test, regressor)

    print("{:.2%} of the data have been classified correctly"
          .format(regressor.score(X_test, Y_test)))


if __name__ == '__main__':
    main()
