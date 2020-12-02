#! /usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse as arg


class myScatter(object):

    def __init__(self, file):
        try:
            self.df = pd.read_csv(file)
            self.df = self.df.drop(
                [
                    'Index',
                    "Defense Against the Dark Arts",
                    "Arithmancy",
                    "Care of Magical Creatures"
                ],
                axis=1)
        except (FileNotFoundError, ValueError, KeyError):
            print("file not found or corrupted")
            self.df = None

    def print(self):
        if self.df is None:
            return

        p = sns.pairplot(
            self.df,
            dropna=True,
            hue="Hogwarts House",
            hue_order=['Ravenclaw', 'Hufflepuff', 'Slytherin', 'Gryffindor']
        )
        p.fig.suptitle("Quelles caractéristiques allez-vous utiliser pour \
entraîner votre prochaine régression logistique ?", y=1, fontsize=16)
        plt.show()

    def run(self):
        self.print()


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="""
    Quelles caractéristiques allez-vous utiliser pour entraîner votre \
prochaine régression logistique ?
    """)
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
