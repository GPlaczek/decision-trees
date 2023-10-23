#!/usr/bin/env python3

from .tree import make_tree, print_tree, map_age_to_labels
import pandas as pd


def main():
    print("Hello world")
    data = pd.read_csv('res/titanic-homework.csv')
    target = 'Survived'
    features = ['Pclass', 'Sex', 'SibSp', 'Parch','Age']
    data['Age']=data['Age'].apply(map_age_to_labels)
    linear_featers=['Age']

    print_tree(make_tree(data,target,features))
    

if __name__ == "__main__":
    main()
