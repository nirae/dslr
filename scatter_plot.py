#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import numpy as np
import argparse as arg

class myScatter(object):

    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.courses = self.df.iloc[:, 6:].columns.to_list()
        self.houses = {}
        self.houses['Gryffindor'] = self.df.loc[self.df['Hogwarts House'] == "Gryffindor"]
        self.houses['Ravenclaw'] = self.df.loc[self.df['Hogwarts House'] == "Ravenclaw"]
        self.houses['Slytherin'] = self.df.loc[self.df['Hogwarts House'] == "Slytherin"]
        self.houses['Hufflepuff'] = self.df.loc[self.df['Hogwarts House'] == "Hufflepuff"]

    def print_grouped(self):
        nrows = 4
        ncols = 4
        fig, axs = plt.subplots(nrows, ncols)
        fig.tight_layout()
        axs[1,3].set_axis_off()
        axs[2,3].set_axis_off()
        axs[3,3].set_axis_off()
        
        def update_i(i, nrows):
            if i[0] + 1 > nrows - 1:
                i[0] = 0
                i[1] += 1
            else:
                i[0] += 1

        i = [0, 0]
        for course in self.courses:
            for key, house in self.houses.items():
                axs[i[0]][i[1]].scatter(house[course], range(len(house[course])), alpha=0.3, label=key)        
            axs[i[0]][i[1]].set_title(course)
            axs[i[0]][i[1]].set_xlabel('grads')
            update_i(i, nrows)
        fig.legend(self.houses, loc='lower right')
        plt.show()
    
    def run(self):
        self.print_grouped()

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
