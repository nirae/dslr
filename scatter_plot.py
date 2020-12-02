#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse as arg


class myScatter(object):

    def __init__(self, file):
        try:
            self.df = pd.read_csv(file)
            self.courses = self.df.iloc[:, 6:].columns.to_list()
            self.houses = {}
            self.houses['Gryffindor'] = \
                self.df.loc[self.df['Hogwarts House'] == "Gryffindor"]
            self.houses['Ravenclaw'] = \
                self.df.loc[self.df['Hogwarts House'] == "Ravenclaw"]
            self.houses['Slytherin'] = \
                self.df.loc[self.df['Hogwarts House'] == "Slytherin"]
            self.houses['Hufflepuff'] = \
                self.df.loc[self.df['Hogwarts House'] == "Hufflepuff"]
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
            for key, house in self.houses.items():
                axs[i[0]][i[1]].scatter(
                    house[course],
                    range(len(house[course])),
                    alpha=0.3,
                    label=key
                )
            axs[i[0]][i[1]].set_title(course)
            axs[i[0]][i[1]].set_xlabel('grads')
            i = self.get_next_axe(axs, i, nrows=nrows)
        fig.suptitle(
            "Quelles sont les deux features qui sont semblables ?",
            y=1,
            fontsize=16
        )
        fig.legend(self.houses, loc='lower right')
        plt.show()

    def run(self):
        self.print_grouped()


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="""
    Quelles sont les deux features qui sont semblables ?
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
