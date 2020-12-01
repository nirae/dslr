#! /usr/bin/env python3

from sklearn.metrics import accuracy_score
import pandas as pd

true = pd.read_csv("datasets/dataset_truth.csv")
predictions = pd.read_csv("houses.csv")
print("modele performance:", accuracy_score(
        true['Hogwarts House'],
        predictions['Hogwarts House']
    ))
