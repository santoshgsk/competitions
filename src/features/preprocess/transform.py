

import numpy as np

def log_transform(data, features = None, skew_thresh=1.4):
    '''

    :param data: The dataframe with all features included
    :param features: A set of features that need to be transformed
    :param skew_thresh: the lower threshold of skeweness of data
    :return: The dataframe with the relevant features log transformed
    '''

    if features is None:
        feats_skewed = data.columns[(data.min() > 0) & (data.skew() > skew_thresh)]
    else:
        feats_skewed = features

    data_copy = data.copy()
    for feat in feats_skewed:
        data_copy[feat] = np.log(data[feat])

    return data_copy, feats_skewed
