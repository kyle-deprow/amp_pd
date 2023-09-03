import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import lightgbm as lgb

from amp_pd.utils.util import (
    create_folds_mapping,
    split_df,
    run_single_fit,
    single_smape1p,
    opt_smape1p,
    run_train,
    run_test
)

class BaseModel:
    def fit(self, df_train):
        raise "NotImplemented"

    def predict(self, df_valid):
        raise "NotImplemented"

    def predict_proba(self, df_valid):
        raise "NotImplemented"

    def fit_predict(self, df_train, df_valid):
        self.fit(df_train)
        return self.predict(df_valid)

    def fit_predict_proba(self, df_train, df_valid):
        self.fit(df_train)
        return self.predict_proba(df_valid)

    def cv(self, sample, sup_sample=None, n_folds=5, random_state=42):
        folds_mapping = create_folds_mapping(sample, n_folds, random_state)

        res = None
        for fold_id in sorted(folds_mapping.unique()):
            df_train, df_valid = split_df(sample, folds_mapping, fold_id)
            if sup_sample is not None:
                df_train = pd.concat([df_train, sup_sample], axis=0)
            p = self.fit_predict(df_train, df_valid)
            delta = pd.DataFrame({"fold": fold_id,  \
                    "patient_id":df_valid["patient_id"], "visit_month":df_valid["visit_month"], \
                    "target_month":df_valid["target_month"], "target_i":df_valid["target_i"], \
                    "target":df_valid["target"], "preds":p})
            res = pd.concat([res, delta], axis=0)

        return res

    def cvx(self, sample, sup_sample=None, n_runs=1, n_folds=5, random_state=42, probs=False):
        np.random.seed(random_state)
        seeds = np.random.randint(0, 1e6, n_runs)

        run_args = []
        for seed in seeds:
            folds_mapping = create_folds_mapping(sample, n_folds, seed)
            for fold_id in sorted(folds_mapping.unique()):
                df_train, df_valid = split_df(sample, folds_mapping, fold_id)
                if sup_sample is not None:
                    df_train = pd.concat([df_train, sup_sample], axis=0)
                run_args.append(dict(
                    df_train = df_train,
                    df_valid = df_valid,
                    fold_id = fold_id,
                    seed = seed,
                    probs = probs
                ))

        res = Parallel(-1)(delayed(run_single_fit)(self, **args) for args in run_args)
        #res = [run_single_fit(self, **args) for args in run_args]
        return pd.concat(res, axis=0)

    def loo(self, sample, sup_sample=None, probs=False, sample2=None):
        if sample2 is None:
            sample2 = sample
        run_args = []
        for patient_id in sample["patient_id"].unique():
            df_train = sample[sample["patient_id"] != patient_id]
            df_valid = sample2[sample2["patient_id"] == patient_id]
            if sup_sample is not None:
                df_train = pd.concat([df_train, sup_sample], axis=0)
            run_args.append(dict(
                df_train = df_train,
                df_valid = df_valid,
                fold_id = None,
                seed = None,
                probs=probs
            ))

        res = Parallel(-1)(delayed(run_single_fit)(self, **args) for args in run_args)
        return pd.concat(res, axis=0)

class LGBClassModel1(BaseModel):
    def __init__(self, params, features) -> None:
        self.params = params
        self.features = features
    
    def fit(self, df_train):
        if self.features is None:
            self.features = [col for col in df_train.columns if col.startswith("v_")]
        lgb_train = lgb.Dataset(df_train[self.features], df_train["target"])
        params0 = {k:v for k,v in self.params.items() if k not in ["n_estimators"]}
        self.m_gbm = lgb.train(params0, lgb_train, num_boost_round=self.params["n_estimators"])
        return self

    def predict_proba(self, df_valid):
        return self.m_gbm.predict(df_valid[self.features])

    def predict(self, df_valid):
        return opt_smape1p(self.predict_proba(df_valid))

class NNRegModel1(BaseModel):
    def __init__(self, cfg, features=None) -> None:
        self.cfg = cfg
        #self.features = features
    
    def fit(self, df_train):
        self.models = [run_train(self.cfg, df_train, None, None, verbose=False) for _ in range(self.cfg.bag_size)]
        return self

    def predict(self, df_valid):
        preds = np.vstack([run_test(model, self.cfg, df_valid) for model in self.models])
        if self.cfg.bag_agg_function == "max":
            return np.max(preds, axis=0)
        elif self.cfg.bag_agg_function == "median":
            return np.median(preds, axis=0)
        else:
            return np.mean(preds, axis=0)