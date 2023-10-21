import sys

import pandas as pd
import numpy as np

def __log_helper(s: pd.Series, group_by: str) -> float:
    size = s.shape[0]
    probs = s.groupby(group_by).size().apply(lambda x: np.log2(x / size) * x / size)
    return - np.sum(probs)

def entropy(s: pd.Series, crit: str = "Survived") -> float:
    return __log_helper(s, crit)

def cond_entropy(s: pd.Series, attr: str, crit: str = "Survived") -> float:
    grouped = s.groupby(attr)
    size = s.shape[0]

    sizes = grouped.size()
    sizes.name = "Size"
    entropies = grouped.apply(lambda x: entropy(x, crit))
    entropies.name = "Entropy"

    return np.sum(pd.concat([entropies, sizes], axis=1).apply(lambda x: x["Size"] / size * x["Entropy"], axis=1))

def gain(s: pd.Series, attr: str, crit: str = "Survived") -> float:
    return entropy(s, crit) - cond_entropy(s, attr, crit)

def intrinsic_info(s: pd.Series, attr: str) -> float:
    return __log_helper(s, attr)

def gain_ratio(s: pd.Series, attr: str, crit: str = "Survived") -> float:
    return gain(s, attr, crit) / intrinsic_info(s, attr)
