import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection import KFold

def split_df(df, folds_mapping, fold_id:int = 0):
    folds = df["patient_id"].map(folds_mapping)

    df_train = df[folds != fold_id]
    df_train = df_train[~df_train["target"].isnull()].reset_index(drop=True)

    df_valid = df[folds == fold_id]
    df_valid = df_valid[~df_valid["target"].isnull()].reset_index(drop=True)
    
    return df_train, df_valid

def create_folds_mapping(df, n_folds=5, random_state=42):
    folds_df = pd.DataFrame({"patient_id":df["patient_id"].unique()})
    folds_df["fold"] = -1

    for i, (_, test_index) in enumerate(KFold(n_splits=n_folds, 
            shuffle=True, random_state=random_state).split(folds_df)):
        folds_df.loc[test_index, "fold"] = i
    folds_mapping = folds_df.set_index(["patient_id"])["fold"]
    return folds_mapping

def smape1p_ind(A, F):
    val = 200 * np.abs(F - A) / (np.abs(A+1) + np.abs(F+1))
    return val

def smape1p(A, F):
    return smape1p_ind(A, F).mean()

def max_dif(val, lst):
    lst0 = [x for x in lst if x < val]
    if len(lst0) == 0:
        return -1
    return val - max(lst0)

def count_prev_visits(val, lst):
    lst0 = [x for x in lst if x < val]
    return len(lst0)