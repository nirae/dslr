#! /usr/bin/env python3

import pandas as pd
import math
import argparse as arg


class myDescribe(object):

    def __init__(self, file):
        try:
            self.data = pd.read_csv(file)
            self.data = self.data.drop(
                [
                    'Hogwarts House',
                    'First Name',
                    'Last Name',
                    'Birthday',
                    'Best Hand'
                ],
                axis=1
            )
        except (FileNotFoundError, ValueError, KeyError):
            print("file not found or corrupted")
            self.data = None
        pd.options.display.float_format = "{:.5f}".format

    def isNaN(self, num):
        return num != num

    def mean(self, iterable):
        """
        Mean / Moyenne
        All column values mean
        formula:
            mean(x) = sum(x) / len(x)
        """

        sum = 0.0
        length = len(iterable)
        for i in iterable:
            if math.isnan(i):
                length -= 1
                continue
            sum += i
        return sum / length

    def count(self, iterable):
        """
        Count
        The number of valid values in the column
        """

        result = 0.0
        for i in iterable:
            if math.isnan(i):
                continue
            result += 1
        return result

    def min(self, iterable):
        """
        Min
        The minimal value in the column
        """

        result = iterable[0]
        for i in iterable:
            if math.isnan(i):
                continue
            if i < result or math.isnan(result):
                result = i
        return result

    def max(self, iterable):
        """
        Max
        The maximal value in the column
        """

        result = iterable[0]
        for i in iterable:
            if math.isnan(i):
                continue
            if i > result or math.isnan(result):
                result = i
        return result

    def percent(self, iterable, percent):
        """
        Percentile / Quartile (25%, 50%, 75%)
        Get the column value at the x% position
        example:
            column = range(1000) -> [0, 1, 2, ... 1000]
            percent = 0.5
            result = column[len(column) * percent] -> 500
            50% value of column is 500, 25% -> 250, etc...
        """

        iterable = [x for x in iterable if not math.isnan(x)]
        iterable.sort()
        i = float(len(iterable) * percent)
        return iterable[int(i)]

    def std(self, iterable):
        """
        Standard deviation / Ecart type
        All column values standard deviation

        The standard deviation is a measure of the dispersion around the mean \
        of a set of values. The smaller the std, the more homogeneous the \
        values are

        https://fr.wikipedia.org/wiki/%C3%89cart_type

        formula:
            std(x) = sum((x - mean) ** 2) / count(x)
        """

        result = 0.0
        mean = self.mean(iterable)
        count = self.count(iterable) - 1
        sum = 0.0
        for i in iterable:
            if math.isnan(i):
                continue
            sum += (i - mean) ** 2
        result = sum / count
        return result ** (0.5)

    def describe(self):
        if self.data is None:
            return None
        index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        data = self.data.select_dtypes(include=['int64', 'float64'])
        result = pd.DataFrame(columns=data.columns.tolist(), index=index)
        for i in result:
            if self.data[i].dtype != 'int64' \
                    and self.data[i].dtype != 'float64':
                continue
            result[i]['mean'] = self.mean(self.data[i].values)
            result[i]['count'] = self.count(self.data[i].values)
            result[i]['std'] = self.std(self.data[i].values)
            result[i]['min'] = self.min(self.data[i].values)
            result[i]['max'] = self.max(self.data[i].values)
            result[i]['50%'] = self.percent(self.data[i].values, 0.5)
            result[i]['25%'] = self.percent(self.data[i].values, 0.25)
            result[i]['75%'] = self.percent(self.data[i].values, 0.75)
        result.style.format({'Index': '{:.5f}'})
        return result


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="describe some data")
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default='datasets/dataset_train.csv',
        help="input data file"
    )
    args = parser.parse_args()

    d = myDescribe(args.file)
    result = d.describe()
    if result is not None:
        print(d.describe())
