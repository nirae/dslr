#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse as arg
import seaborn as sns


class Predicter(object):

    def __init__(self, thetafile, datafile, outfile):
        self.houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        self.output = outfile

        try:
            df = pd.read_csv(thetafile)
            self.thetas = df.drop(df.columns[0], axis=1)
        except FileNotFoundError:
            print("file %s not found" % thetafile)
            self.thetas = None

        try:
            self.data = pd.read_csv(datafile)
            self.data = self.data.iloc[:, 6:]
            self.data = self.data.replace(np.nan, 0)
            self.data = self.standardize(self.data)
        except FileNotFoundError:
            print("file %s not found" % datafile)
            self.data = None

    def standardize(self, data):
        """
        Standard Scaler
        same as sklearn.preprocessing.StandardScaler
        formula:
            z = (x - mean) / std
        """

        return (data - data.mean()) / data.std()

    def sigmoid(self, x):
        """
        Modele function / sigmoid function
        formula:
            e = Euler number / Neper constant = 2,71828

            f(x) = 1 / (1 + e ** -x)
        """

        return 1 / (1 + np.exp(-x))

    def save_predictions(self):
        """
        Save the predictions in a csv file. Default: houses.csv
        """

        self.predictions.index.name = "Index"
        self.predictions.to_csv(self.output)
        print("the predictions was written in the file %s" % self.output)

    def get_next_axe(self, axes, current, nrows):
        if current[0] + 1 > nrows - 1:
            current[0] = 0
            current[1] += 1
        else:
            current[0] += 1
        return current

    def show(self):
        """
        Show the predictions for each houses in a plot
        """

        nrows = 2
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols)
        fig.tight_layout()

        palette = sns.color_palette()
        houses_order = ['Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor']
        colors = {house: palette[i] for i, house in enumerate(houses_order)}

        ax = [0, 0]

        sorted_data = self.data.sort_values(
            self.data.columns.values.tolist(),
            ascending=False
        )

        for i, row in self.thetas.iterrows():
            sig = self.sigmoid(sorted_data.dot(row))
            axs[ax[0]][ax[1]].plot(
                sorted_data.dot(row),
                sig,
                '.',
                alpha=0.5,
                label=self.houses[i]
            )
            axs[ax[0]][ax[1]].set_title(
                self.houses[i],
                c=colors[self.houses[i]]
            )
            ax = self.get_next_axe(axs, ax, nrows=nrows)
        plt.show()

    def run(self, show=False):
        if self.thetas is None or self.data is None:
            return None

        result = {}
        for i, row in self.thetas.iterrows():
            sig = self.sigmoid(self.data.dot(row))
            result[self.houses[i]] = sig
        result = pd.DataFrame(result)
        if show:
            print(result)
        self.predictions = pd.DataFrame(
            [self.houses[r.argmax()] for _, r in result.iterrows()],
            columns=['Hogwarts House']
        )
        if show:
            self.show()
        return self.predictions


if __name__ == '__main__':
    parser = arg.ArgumentParser(description="""
    Logistic Regression predicter program
    Using a trained thetas csv file, generated with the logreg_train program
    """)
    parser.add_argument(
        '-d',
        '--datafile',
        type=str,
        default='datasets/dataset_test.csv',
        help="input data file"
    )
    parser.add_argument(
        '-t',
        '--thetafile',
        type=str,
        default='thetas.csv',
        help="output theta file"
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='houses.csv',
        help="output houses file"
    )
    parser.add_argument(
        '-s',
        '--show',
        action="store_true",
        default=False,
        help="show logistic regression"
    )
    args = parser.parse_args()
    predicter = Predicter(
        datafile=args.datafile,
        thetafile=args.thetafile,
        outfile=args.output
    )
    result = predicter.run(show=args.show)
    if result is not None:
        predicter.save_predictions()
    else:
        print("prediction failed")
