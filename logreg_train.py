#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse as arg
from progress.bar import ChargingBar
import seaborn as sns


class Trainer(object):

    def __init__(self, datafile, outfile, learning_rate=0.1, train_range=1000):
        self.history = {}
        self.datafile = datafile
        self.output = outfile
        self.df = pd.read_csv(datafile)
        self.learning_rate = learning_rate
        self.range = train_range
        self.houses = {
            'Ravenclaw': 0,
            'Slytherin': 1,
            'Gryffindor': 2,
            'Hufflepuff': 3
        }
        self.houses_indexes = \
            [self.houses[x] for x in self.df['Hogwarts House']]
        self.data = self.df.iloc[:, 6:]
        self.data = self.data.replace(np.nan, 0)
        self.data = self.standardize(self.data)
        self.thetas = []

    def standardize(self, data):
        return (data - data.mean()) / data.std()

    def sigmoid(self, x):
        """
        Modele function / sigmoid function
        formula:
            e = Euler number / Neper constant = 2,71828

            g(x) = 1 / (1 + e ** -x)
        """
        return 1 / (1 + np.exp(-x))

    def cost(self, theta, y):
        """
        Cost function and cost derivative

        cost formula:
            h(x) = g(theta @ x)
            -(1 / m) * (SUM((y * LOG(h(x))) + ((1 - T) * (LOG(1 - h(x))))))
        cost derivative:
            (1 / m) * ((sig - y.T) @ x))

        m = len(x)
        g = sigmoid function
        """

        m = len(self.data)
        sig = self.sigmoid(np.dot(theta, self.data.T))
        cost = (np.sum((y.T * np.log(sig)) + ((1 - y.T) * (np.log(1 - sig))))) / -m
        derivative = (np.dot((sig - y.T), self.data)) / m
        return derivative, cost

    def gradient_descent(self, y):
        """
        Gradient descent to get the correct theta

        formula:

        range = numbers of iterations
        lr = learning rate, determine the "size of the step"

        theta = 0
        while range
            theta = theta - (lr * cost_derivative)

        The goal is to find the good learning rate and range
        default:

            learning rate = 0.1
            range = 2000
        """

        history = []
        theta = np.zeros((1, self.data.shape[1]))
        for _ in range(self.range):
            self.bar.next()
            derivative, cost = self.cost(theta, y)
            theta -= self.learning_rate * derivative
            history.append(cost)
        return theta[0].tolist(), history

    def save_thetas(self):
        """
        Save the thetas in a csv file. Default: thetas.csv
        """

        self.thetas.to_csv(self.output)

    def get_next_axe(self, axes, current, nrows):
        if current[0] + 1 > nrows - 1:
            current[0] = 0
            current[1] += 1
        else:
            current[0] += 1
        return current

    def show_history(self):
        """
        Show the history of cost for each houses in a plot
        """

        nrows = 2
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols)
        fig.tight_layout()

        palette = sns.color_palette()
        houses_order = ['Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor']
        colors = {house: palette[i] for i, house in enumerate(houses_order)}

        ax = [0, 0]
        for key, val in self.history.items():
            axs[ax[0]][ax[1]].plot(val, label=key)
            axs[ax[0]][ax[1]].set_title(key, c=colors[key])
            ax = self.get_next_axe(axs, ax, nrows=nrows)
        plt.show()

    def train(self, show_history):
        self.bar = ChargingBar(
            'Training',
            max=(self.range * len(self.houses)),
            suffix='%(percent)d%% - eta %(eta)ds'
        )
        for i, val in enumerate(self.houses):
            y = []
            for house in self.houses_indexes:
                y.append(1 if house == i else 0)
            theta, history = self.gradient_descent(np.asarray(y))
            self.history[val] = history
            self.thetas.append(theta)
        self.bar.finish()
        self.thetas = pd.DataFrame(
            self.thetas,
            columns=self.data.columns,
            index=self.houses
        )
        if show_history:
            self.show_history()
        return self.thetas


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="""
    Logistic Regression train program, using gradient descent
    """)
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='datasets/dataset_train.csv',
        help="input data file"
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='thetas.csv',
        help="output thetas file"
    )
    parser.add_argument(
        '-H',
        '--history',
        action="store_true",
        default=False,
        help="show history plot"
    )
    parser.add_argument(
        '-l',
        '--learningrate',
        type=float,
        default=0.1,
        help="learning rate for training"
    )
    parser.add_argument(
        '-r',
        '--range',
        type=int,
        default=2000,
        help="training range for training (epochs)"
    )
    args = parser.parse_args()

    trainer = Trainer(
        args.input,
        args.output,
        learning_rate=args.learningrate,
        train_range=args.range
    )
    thetas = trainer.train(show_history=args.history)
    print("Thetas:\n", thetas)
    trainer.save_thetas()
