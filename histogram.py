#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse as arg

class myHistogram(object):

    def __init__(self, file):
        df = pd.read_csv(file)
        self.courses = df.iloc[:, 6:].columns.to_list()
        self.houses = {}
        self.houses['Gryffindor'] = df.loc[df['Hogwarts House'] == "Gryffindor"]
        self.houses['Ravenclaw'] = df.loc[df['Hogwarts House'] == "Ravenclaw"]
        self.houses['Slytherin'] = df.loc[df['Hogwarts House'] == "Slytherin"]
        self.houses['Hufflepuff'] = df.loc[df['Hogwarts House'] == "Hufflepuff"]

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
            for house, house_val in self.houses.items():
                axs[i[0]][i[1]].hist(house_val[course], bins=20, alpha=0.5, histtype='stepfilled', label=house)        
            axs[i[0]][i[1]].set_title(course)
            axs[i[0]][i[1]].set_xlabel('grads')
            update_i(i, nrows)
        fig.legend(self.houses, loc='lower right')
        plt.show()
    
    def run(self, grouped=False):
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
    h = myHistogram(args.file)
    h.run()
