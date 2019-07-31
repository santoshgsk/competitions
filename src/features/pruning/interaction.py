
import operator
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

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


def model_select_all_thresh(feature_importance, bst_model, train_x, train_y, model):
    sorted_thresholds = np.sort(feature_importance[1].unique())

    for thresh in sorted_thresholds:
        select_from_model = SelectFromModel(bst_model, threshold=thresh, prefit=True)
        train_x_subset = select_from_model.transform(train_x)

        model.fit(train_x_subset, train_y)

        preds = model.predict_proba(train_x_subset)
        auc_score = roc_auc_score(train_y.values, preds[:, 1])
        print(f'Thresh {thresh} number of features {train_x_subset.shape[1]} auc score {auc_score}')


def model_select_features(model, train_x, train_y, importance_thresh=None, run_all_thresh=False):

    base_model = clone(model)
    bst_model = model.fit(train_x, train_y)
    feature_importance = pd.DataFrame([train_x.columns, bst_model.feature_importances_]).T
    feature_importance.sort_values([1], ascending=False, inplace=True)

    model_selected_features = feature_importance[0].values
    if importance_thresh is not None:
        model_selected_features = feature_importance[feature_importance[1] >= importance_thresh][0].values
    elif run_all_thresh:
        model_select_all_thresh(feature_importance, bst_model, train_x, train_y, base_model)

    return model_selected_features, feature_importance
