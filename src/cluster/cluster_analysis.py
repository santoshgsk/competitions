import pandas as pd
import numpy as np


def get_correlation_coclusters(data, cocluster_model):
    corr_data = data.corr()
    rows_ids = corr_data.index
    corr_data_np = corr_data.to_numpy()
    cocluster_model.fit(corr_data)
    clusterd_corr = corr_data_np[np.argsort(cocluster_model.row_labels_)]
    clusterd_corr = clusterd_corr[:, np.argsort(cocluster_model.column_labels_)]
    clusterd_corr_df = pd.DataFrame(clusterd_corr, index=rows_ids[np.argsort(cocluster_model.row_labels_)],
                               columns=rows_ids[np.argsort(cocluster_model.column_labels_)])
    return clusterd_corr_df
