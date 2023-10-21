#!/usr/bin/env python3

from .tree import make_tree
import pandas as pd


def main():
    print("Hello world")
    data = pd.read_csv('res/titanic-homework.csv')
    target = 'Survived'
    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    print(make_tree(data,target,features))

if __name__ == "__main__":
    main()
