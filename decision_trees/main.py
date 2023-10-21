#!/usr/bin/env python3

from .tree import make_tree, print_tree
import pandas as pd


def main():
    print("Hello world")
    data = pd.read_csv('res/titanic-homework.csv')
    target = 'Survived'
    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    linear_featers=['Age']
    print_tree(make_tree(data,target,features,linear_featers))
    

if __name__ == "__main__":
    main()
