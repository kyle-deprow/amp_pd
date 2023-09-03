import os
from joblib import Parallel, delayed
from collections import defaultdict
from copy import copy
import random
from tqdm import tqdm
import gc
from types import SimpleNamespace

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import SequentialSampler, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from scipy.special import softmax
torch.set_num_threads(1)


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

def smape1p_opt(x):
    #return np.median(x)
    tgts = np.arange(0, 61)
    #tgts = [smape(x, val) for val in np.arange(0, 61)]
    scores = [smape1p(x, val) for val in tgts]
    return tgts[np.argmin(scores)]

def run_single_fit(model, df_train, df_valid, fold_id, seed, probs):
    if probs:
        p = model.fit_predict_proba(df_train, df_valid)
        p = pd.DataFrame(p, columns=[f"prob_{i}" for i in range(p.shape[1])]).reset_index(drop=True)
        res = pd.DataFrame({"seed":seed, "fold": fold_id, \
            "patient_id":df_valid["patient_id"], "visit_month":df_valid["visit_month"], \
            "target_month":df_valid["target_month"], "target_i":df_valid["target_i"], \
            "target":df_valid["target"]}).reset_index(drop=True)
        return pd.concat([res, p], axis=1)
    else:
        p = model.fit_predict(df_train, df_valid)
        return pd.DataFrame({"seed":seed, "fold": fold_id, \
            "patient_id":df_valid["patient_id"], "visit_month":df_valid["visit_month"], \
            "target_month":df_valid["target_month"], "target_i":df_valid["target_i"], \
            "target":df_valid["target"], "preds":p})

def single_smape1p(preds, tgt):
    x = np.tile(np.arange(preds.shape[1]), (preds.shape[0], 1))
    x = np.abs(x - tgt) / (2 + x + tgt)
    return (x * preds).sum(axis=1)

def opt_smape1p(preds):
    x = np.hstack([single_smape1p(preds, i).reshape(-1,1) for i in range(preds.shape[1])])
    return x.argmin(axis=1)

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.features = df[cfg.features].values
        if self.mode != "test":
            self.targets = df[self.cfg.target_column].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(df))

    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.targets[idx]
        
        feature_dict = {
            "input": torch.tensor(features),
            "target_norm": torch.tensor(targets),
        }
        return feature_dict

    def __len__(self):
        return len(self.df)

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.n_classes = cfg.n_classes
        self.cnn = nn.Sequential(*([
            nn.Linear(len(self.cfg.features), cfg.n_hidden),
            nn.LeakyReLU(),
            ] +
            [
            nn.Linear(cfg.n_hidden, cfg.n_hidden),
            nn.LeakyReLU(),
            ] * self.cfg.n_layers)
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.n_hidden, self.n_classes),
            nn.LeakyReLU(),
        )

    def forward(self, batch):
        input = batch["input"].float()
        y = batch["target_norm"]
        x = input
        x = self.cnn(x)
        preds = self.head(x).squeeze(-1)
        loss = (torch.abs(y - preds) / (torch.abs(0.01 + y) + torch.abs(0.01 + preds))).mean()
        return {"loss": loss, "preds": preds, "target_norm": y}

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_train_dataloader(train_ds, cfg, verbose):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    if verbose:
        print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg, verbose):
    sampler = SequentialSampler(val_ds)
    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    if verbose:
        print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = False
    #torch.backends.cudnn.benchmark = True

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def run_eval(model, val_dataloader, cfg, pre="val", verbose=True):
    model.eval()
    torch.set_grad_enabled(False)
    val_data = defaultdict(list)
    if verbose:
        progress_bar = tqdm(val_dataloader)
    else:
        progress_bar = val_dataloader
    for data in progress_bar:
        batch = batch_to_device(data, cfg.device)
        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)
        for key, val in output.items():
            val_data[key] += [output[key]]
    for key, val in output.items():
        value = val_data[key]
        if len(value[0].shape) == 0:
            val_data[key] = torch.stack(value)
        else:
            val_data[key] = torch.cat(value, dim=0)

    preds = val_data["preds"].cpu().numpy()
    if (pre == "val") and verbose:
        metric = smape1p(100*val_data["target_norm"].cpu().numpy(), 100*preds)
        print(f"{pre}_metric 1 ", metric)
        metric = smape1p(100*val_data["target_norm"].cpu().numpy(), np.round(100*preds))
        print(f"{pre}_metric 2 ", metric)
    
    return 100*preds

def run_train(cfg, train_df, val_df, test_df=None, verbose=True):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)

    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    if verbose:
        print("seed", cfg.seed)
    set_seed(cfg.seed)

    train_dataset = CustomDataset(train_df, cfg, aug=None, mode="train")
    train_dataloader = get_train_dataloader(train_dataset, cfg, verbose)
    
    if val_df is not None:
        val_dataset = CustomDataset(val_df, cfg, aug=None, mode="val")
        val_dataloader = get_val_dataloader(val_dataset, cfg, verbose)

    if test_df is not None:
        test_dataset = CustomDataset(test_df, cfg, aug=None, mode="test")
        test_dataloader = get_val_dataloader(test_dataset, cfg, verbose)

    model = Net(cfg)
    model.to(cfg.device)

    total_steps = len(train_dataset)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=0)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        set_seed(cfg.seed + epoch)
        if verbose:
            print("EPOCH:", epoch)
            progress_bar = tqdm(range(len(train_dataloader)))
        else:
            progress_bar = range(len(train_dataloader))
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()

        for itr in progress_bar:
            i += 1
            data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            batch = batch_to_device(data, cfg.device)
            if cfg.mixed_precision:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)
            loss = output_dict["loss"]
            losses.append(loss.item())
            if cfg.mixed_precision:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        if val_df is not None:
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                run_eval(model, val_dataloader, cfg, pre="val", verbose=verbose)

    if test_df is not None:
        return run_eval(model, test_dataloader, cfg, pre="test", verbose=verbose)
    else:
        return model

def run_test(model, cfg, test_df):
    test_dataset = CustomDataset(test_df, cfg, aug=None, mode="test")
    test_dataloader = get_val_dataloader(test_dataset, cfg, verbose=False)
    return run_eval(model, test_dataloader, cfg, pre="test", verbose=False)
