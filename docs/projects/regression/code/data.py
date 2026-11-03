import sys
from pathlib import Path

import numpy as np
import pandas as pd


EDA_CODE = Path(__file__).resolve().parents[2] / "eda" / "code"
sys.path.insert(0, str(EDA_CODE))

from exercise4 import CSV, CUT, SEED, FEATURES, TARGET, build_features, make_preprocess, split

VAL_CUT = "2023-01-01"


def load_raw():
    df = pd.read_csv(CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("Timestamp")
    return df


def split_train_val(X_train: pd.DataFrame, y_train: pd.Series):
    x_tr, y_tr = X_train[X_train.index < VAL_CUT], y_train[X_train.index < VAL_CUT]
    x_val, y_val = X_train[X_train.index >= VAL_CUT], y_train[X_train.index >= VAL_CUT]
    x_tr = x_tr.iloc[:-1]
    y_tr = y_tr.iloc[:-1]
    return x_tr, y_tr, x_val, y_val


def load_splits():
    df = load_raw()
    X_train, y_train, X_test, y_test = split(build_features(df))
    x_tr, y_tr, x_val, y_val = split_train_val(X_train, y_train)
    preprocess = make_preprocess()
    X_train_t = preprocess.fit_transform(x_tr)  # fit só no treino
    X_test_t = preprocess.transform(X_test)
    X_val_t = preprocess.transform(x_val)
    return np.asarray(X_train_t, dtype=np.float32), np.asarray(y_tr,dtype=np.float32),np.asarray(X_val_t,dtype=np.float32), np.asarray(y_val,dtype=np.float32), np.asarray(X_test_t,dtype=np.float32), np.asarray(y_test,dtype=np.float32)


def summary(X_tr, y_tr, X_val, y_val, X_te, y_te):
    splits = [("treino", X_tr, y_tr), ("validação", X_val, y_val), ("teste", X_te, y_te)]
    for nome, X, y in splits:
        print(f"\n{nome}: {len(y)} linhas | NaN em X: {np.isnan(X).sum()} | NaN em y: {np.isnan(y).sum()}")
        print(f"  y (alvo): média {y.mean():.3e} | desvio {y.std():.3e}")
        for feature, media, desvio in zip(FEATURES, X.mean(axis=0), X.std(axis=0)):
            print(f"  X[{feature}]: média {media:+.3f} | desvio {desvio:.3f}")



if __name__ == "__main__":
    np.random.seed(SEED)
    summary(*load_splits())
