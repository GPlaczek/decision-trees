from .metrics import entropy, gain_ratio
import pandas as pd
import json
def make_tree(data,target,features):

    if len(features) == 0 or len(data[target].unique()) == 1:
        return int(data[target].value_counts().nlargest(1).index[0])

    best_feature = max(features, key=lambda f: gain_ratio(data, f, target))
    tree = {best_feature: {}}
    features = [f for f in features if f != best_feature]

    # if best_feature not in linear_values:
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][str(value)] = make_tree(subset, target, features)
    #else make divisions

    return tree

def print_tree(tree):
    print( json.dumps(tree,indent=2))