from .metrics import entropy, gain_ratio

import json

import pandas as pd
import random
import graphviz

edges=[]
dot = graphviz.Digraph()


def make_tree(
    data: pd.Series,
    target: str,
    features: list[str],
    linear_featers: list[str] = [],
    idx: str = '0',
    last_best_feature: str = '-1',
    value: str = '0'
) -> dict:
    if len(features) == 0 or len(data[target].unique()) == 1:
        d=data[target].value_counts().nlargest(1).index[0]

        dot.node(f'{last_best_feature}{d}', str(d))
        dot.edge(last_best_feature, f'{last_best_feature}{d}', label=value)

        return d

    best_feature = max(features, key=lambda f: gain_ratio(data, f, target))

    # connections betweens subtrees
    if last_best_feature != '-1':
        dot.node(f'{best_feature}{idx}', best_feature)
        dot.edge(last_best_feature, f'{best_feature}{idx}', label=value)
    else:
        dot.node(f'{best_feature}{idx}',best_feature)

    tree = { best_feature: {} }
    features = [f for f in features if f != best_feature]

    idm = len(data[best_feature].unique())

    if best_feature not in linear_featers:
        for idxs, value in enumerate(data[best_feature].unique()):
            subset = data[data[best_feature] == value]
            tree[best_feature][str(value)] = make_tree(subset, target, features, linear_featers, f'{idx}{idxs}', f'{best_feature}{idx}', str(value))

    # else make divisions

    return tree

def map_age_to_labels(age):
    if 0 <= age <= 20:
        return "young"
    elif 20 < age <= 40:
        return "middle"
    elif 40 < age <= 100:
        return "old"
    else:
        return "unknown"

def print_tree(tree: dict):
    print(tree.keys())
    print(tree.items())
    print(tree)


    dot.render('doctest-output/round-table.gv')
    dot
    print(edges)
    print(len(edges)+1)
