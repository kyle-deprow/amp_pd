import pandas as pd

from amp_pd.data.make_dataset import load_data
from amp_pd.features.build_features import DataPrep
from amp_pd.models.models import LGBClassModel1, NNRegModel1

proteins, peptides, clinical, supplement = load_data("../input/amp-parkinsons-disease-progression-prediction")
dp3 = DataPrep()
dp3.fit(proteins, peptides, clinical)

sample3 = dp3.transform_train(proteins, peptides, clinical)
sample3 = sample3[~sample3["target"].isnull()]
sample3["is_suppl"] = 0

sup_sample3 = dp3.transform_train(proteins, peptides, supplement)
sup_sample3 = sup_sample3[~sup_sample3["target"].isnull()]
sup_sample3["is_suppl"] = 1

params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 87,
        "n_estimators": 300,

        'learning_rate': 0.019673004699536346,
        'num_leaves': 208,
        'max_depth': 14,
        'min_data_in_leaf': 850,
        'feature_fraction': 0.5190632906197453,
        'lambda_l1': 7.405660751699475e-08,
        'lambda_l2': 0.14583961675675494,
        'max_bin': 240,
    
        'verbose': -1,
        'force_col_wise': True,
        'n_jobs': -1,
    }

features = ["target_i", "target_month", "horizon", "visit_month", "visit_6m", "blood_taken"]
features += ["visit_18m", "is_suppl"]
features += ["count_non12_visits"]
features += ["visit_48m"]

model_lgb = LGBClassModel1(params, features)
model_lgb = model_lgb.fit(pd.concat([sample3, sup_sample3], axis=0))

cfg = SimpleNamespace(**{})

cfg.tr_collate_fn = None
cfg.val_collate_fn = None

cfg.target_column = "target_norm"
cfg.output_dir = "results/nn_temp"
cfg.seed = -1
cfg.eval_epochs = 1
cfg.mixed_precision = False
cfg.device = "cpu"

cfg.n_classes = 1
cfg.batch_size = 128
cfg.batch_size_val = 256
cfg.n_hidden = 64
cfg.n_layers = 2 #3
cfg.num_workers = 0
cfg.drop_last = False
cfg.gradient_clip = 1.0

cfg.bag_size = 1
cfg.bag_agg_function = "mean"
cfg.lr = 2e-3
cfg.warmup = 0
cfg.epochs = 10

cfg.features = ["visit_6m"]
cfg.features += [c for c in sample3.columns if c.startswith("t_month_eq_")]
cfg.features += [c for c in sample3.columns if c.startswith("v_month_eq_")]
cfg.features += [c for c in sample3.columns if c.startswith("hor_eq_")]
cfg.features += [c for c in sample3.columns if c.startswith("target_n_")]
cfg.features += ["visit_18m"]
cfg.features += ["visit_48m"]
cfg.features += ["is_suppl"]
cfg.features += ["horizon_scaled"]

model_nn = NNRegModel1(cfg)
model_nn = model_nn.fit(pd.concat([sample3, sup_sample3], axis=0))
