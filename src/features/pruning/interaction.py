
import operator
import numpy as np
from collections import defaultdict

def get_correlated_features(data, thresh):

    corr_mat = data.corr()

    similar_feats = defaultdict(lambda: list)
    for c in data.columns:
        corr_values = corr_mat[c]
        corr_feats = list(corr_values.index[
                              (corr_values > np.abs(thresh)) | (corr_values < - np.abs(thresh) )])
        sim_feats = list(set(corr_feats) - set([c]))
        if len(sim_feats) > 0:
            similar_feats[c] = sim_feats

    similar_feats_cnt = defaultdict()
    for k, v in similar_feats.items():
        similar_feats_cnt[k] = len(v)

    sorted_feats_corr = sorted(similar_feats_cnt.items(), key=operator.itemgetter(1), reverse=True)

    feats_to_delete = []
    for t in sorted_feats_corr:
        feat = t[0]
        if len(list(set(similar_feats[feat]) - set(feats_to_delete))) > 0:
            feats_to_delete.append(feat)

    return feats_to_delete