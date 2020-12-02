#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse as arg


class myHistogram(object):

    def __init__(self, file):
        try:
            df = pd.read_csv(file)
            self.courses = df.iloc[:, 6:].columns.to_list()
            self.houses = {}
            self.houses['Gryffindor'] = \
                df.loc[df['Hogwarts House'] == "Gryffindor"]
            self.houses['Ravenclaw'] = \
                df.loc[df['Hogwarts House'] == "Ravenclaw"]
            self.houses['Slytherin'] = \
                df.loc[df['Hogwarts House'] == "Slytherin"]
            self.houses['Hufflepuff'] = \
                df.loc[df['Hogwarts House'] == "Hufflepuff"]
        except (FileNotFoundError, ValueError, KeyError):
            print("file not found or corrupted")
            self.houses = None
            self.courses = None

    def get_next_axe(self, axes, current, nrows):
        if current[0] + 1 > nrows - 1:
            current[0] = 0
            current[1] += 1
        else:
            current[0] += 1
        return current

    def print_grouped(self):
        if self.courses is None or self.houses is None:
            return

        nrows = 4
        ncols = 4
        fig, axs = plt.subplots(nrows, ncols)
        fig.tight_layout()
        axs[1, 3].set_axis_off()
        axs[2, 3].set_axis_off()
        axs[3, 3].set_axis_off()

        i = [0, 0]
        for course in self.courses:
            for house, house_val in self.houses.items():
                axs[i[0]][i[1]].hist(
                    house_val[course],
                    bins=20,
                    alpha=0.5,
                    histtype='stepfilled',
                    label=house
                )
            axs[i[0]][i[1]].set_title(course)
            axs[i[0]][i[1]].set_xlabel('grads')
            i = self.get_next_axe(axs, i, nrows=nrows)
        fig.suptitle("Quel cours de Poudlard a une répartition des notes \
homogènes entre les quatres maisons ?", y=1, fontsize=16)
        fig.legend(self.houses, loc='lower right')
        plt.show()

    def run(self):
        self.print_grouped()


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="""
        Quel cours de Poudlard a une répartition des notes homogènes entre \
        les quatres maisons ?
    """)
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default='datasets/dataset_train.csv',
        help="input data file"
    )
    args = parser.parse_args()
    h = myHistogram(args.file)
    h.run()
