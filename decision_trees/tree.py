from .metrics import entropy, gain_ratio
import pandas as pd
import random
import json
import graphviz
    
edges=[]
dot = graphviz.Digraph()

def make_tree(data,target,features,linear_featers,idx='0',last_best_feature_value='-1'):

    # print(last_best_feature_value)
    # print(set(data[target]))
    # print(len(set(data[target])))
    if len(features) == 0 or len(set(data[target])) == 1:
        d=data[target].value_counts().nlargest(1).index[0]
        dot.node(last_best_feature_value + str(d), str(d))
        dot.edge(last_best_feature_value, last_best_feature_value +str(d))
        return d
        
        

    best_feature = max(features, key=lambda f: gain_ratio(data, f, target))
    # connections betweens subtrees
    if last_best_feature_value != '-1':
        dot.node(str(best_feature)+idx,best_feature)
        dot.edge(last_best_feature_value,str(best_feature) + idx)
    else:
        dot.node(str(best_feature)+idx,best_feature)
    tree = {best_feature: {}}
    features = [f for f in features if f != best_feature]

    idm = len(data[best_feature].unique())

    if best_feature not in linear_featers:
        for idxs,value in enumerate(data[best_feature].unique()):
            dot.node(str(str(value)+'_' + idx+str(idxs)),str(value))
            dot.edge(str(best_feature) + idx,str(value)+'_' + idx+str(idxs))
            subset = data[data[best_feature] == value]
            tree[best_feature][str(value)] = make_tree(subset, target, features,linear_featers,idx+str(idxs),str(value)+'_' + idx+str(idxs))
    # else make divisions

    return tree
    


def print_tree(tree):
    # print( json.dumps(tree)
    print(tree.keys())
    print(tree.items())
    print(tree)


    dot.render('doctest-output/round-table.gv')
    dot
    print(edges)
    print(len(edges)+1)

    
    
    