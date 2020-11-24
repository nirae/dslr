#! /usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import numpy as np
import argparse as arg
from datetime import datetime


class myScatter(object):

    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.df = self.df.drop(['Index', "Defense Against the Dark Arts", "Arithmancy", "Care of Magical Creatures"], axis=1)
        # self.df = self.df.drop(["Defense Against the Dark Arts", "Arithmancy", "Care of Magical Creatures"], axis=1)
        # self.courses = self.df.iloc[:, 6:].columns.to_list()
        # self.df['Best Hand'] = self.df['Best Hand'].map({'Right': 0, 'Left': 1})
        # f = lambda x: datetime.strptime(x, "%Y-%m-%d")
        # self.df['Birthday'] = self.df['Birthday'].map(f)

    def print(self):
        # sns.pairplot(self.df, hue="Hogwarts House", hue_order=['Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor'], y_vars=['Index', 'Best Hand', 'Birthday'], x_vars=self.courses)
        sns.pairplot(self.df, dropna=True, hue="Hogwarts House", hue_order=['Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor'])
        plt.show()
    
    def run(self):
        self.print()


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="Quel cours de Poudlard a une répartition des notes homogènes entre les quatres maisons ?")
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default='datasets/dataset_train.csv',
        help="input data file"
    )
    args = parser.parse_args()
    s = myScatter(args.file)
    s.run()
