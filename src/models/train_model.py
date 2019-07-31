
from collections import defaultdict
import xgboost as xgb
import pandas as pd

def train_xgb_cv(train_x, train_y, test, cvsplit, param, num_round=130):
    test_x = xgb.DMatrix(test)
    full_train_x = xgb.DMatrix(train_x)

    test_preds = None
    train_preds = None

    count = 0
    for train_index, val_index in cvsplit.split(train_x, train_y):
        x_train, x_val = train_x.iloc[train_index], train_x.iloc[val_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[val_index]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        evallist = [(dval, 'eval'), (dtrain, 'train')]

        bst = xgb.train(param, dtrain, num_round, evallist)

        pred_test = bst.predict(test_x)
        if test_preds is None:
            test_preds = pred_test
        else:
            test_preds *= pred_test

        pred_train = bst.predict(full_train_x)
        if train_preds is None:
            train_preds = pred_train
        else:
            train_preds *= pred_train

        count += 1

    test_preds = test_preds ** (1.0 / count)
    train_preds = train_preds ** (1.0 / count)

    return train_preds, test_preds


def baseline_train(data_dict, model, feature_importances=False):
    train_x = data_dict['train_x']
    train_y = data_dict['train_y']

    model.fit(train_x, train_y)

    return_dict = defaultdict()
    return_dict['train_preds'] = model.predict()

    if 'val_x' in data_dict:
        val_x = data_dict['val_x']
        val_preds = model.predict(val_x)
        return_dict['val_preds'] = val_preds

    if feature_importances:
        fi = pd.DataFrame([train_x.columns, model.feature_importances_]).T.sort_values([1], ascending=False)
        return_dict['feature_importances'] = fi

    return return_dict


