from .metrics import entropy, gain_ratio
import pandas as pd
def make_tree(data,target,features):
   
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].value_counts().nlargest(1).index[0]

    best_feature = max(features, key=lambda f: gain_ratio(data, f, target))
    tree = {best_feature: {}}
    features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = make_tree(subset, target, features)

    return tree

