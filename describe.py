#! /usr/bin/env python3

import pandas as pd
import math
import argparse as arg


class myDescribe(object):

    def __init__(self, file):
        self.data = pd.read_csv(file)
        pd.options.display.float_format = "{:.5f}".format

    def isNaN(self, num):
        return num != num

    def mean(self, iterable):
        s = 0.0
        length = len(iterable)
        for i in iterable:
            if math.isnan(i):
                length -= 1
                continue
            s += i
        return s / length

    def count(self, iterable):
        result = 0.0
        for i in iterable:
            if math.isnan(i):
                continue
            result += 1
        return result

    def min(self, iterable):
        result = iterable[0]
        for i in iterable:
            if math.isnan(i):
                continue
            if i < result or math.isnan(result):
                result = i
        return result

    def max(self, iterable):
        result = iterable[0]
        for i in iterable:
            if math.isnan(i):
                continue
            if i > result or math.isnan(result):
                result = i
        return result

    def percent(self, iterable, percent):
        iterable = [x for x in iterable if not math.isnan(x)]
        iterable.sort()
        i = float(len(iterable) * percent)
        return iterable[int(i)]

    def std(self, iterable):
        result = 0.0
        mean = self.mean(iterable)
        count = self.count(iterable) - 1
        s = 0.0
        for i in iterable:
            if math.isnan(i):
                continue
            s += (i - mean) ** 2
        result = s / count
        return result ** (0.5)

    def describe(self):
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
    print(d.describe())
