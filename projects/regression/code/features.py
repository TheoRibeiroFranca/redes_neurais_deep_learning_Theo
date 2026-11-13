# Monta as matrizes F4 e F65 da regressão e grava em cache (data/cache/*.npz).
# - F4: as 4 features do EDA, pré-processadas com make_preprocess() do exercise4.py.
# - F65: as 4 features (volume em log1p) nos lags 0..15 + rv_60 (desvio de ret_1 em [t-59, t]).
# Escolhas e aulas:
# - Aula 2, Data Leakage §3: toda janela é só para trás (lags e rolling sem center); o alvo é t+1.
# - Aula 2, Train/Val/Test Split §1: o split é uma data (2023-01-01 e 2024-01-01), via data.py.
# - Aula 2, Data Leakage §4 e Scaling: StandardScaler ajustado só nas linhas de tr; val e teste
#   são só transformados.
# - Lags e rv_60 são calculados na série inteira antes do split: só olham para o passado,
#   e a última linha de cada parte de treino continua descartada (o alvo não cruza o corte).
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_raw, split_train_val  # noqa: E402  (data.py põe o EDA no path)
from exercise4 import FEATURES, TARGET, build_features, make_preprocess, split  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "data" / "cache"
N_LAGS = 15          # lags 0..15
RV_WINDOW = 60       # rv_60: linhas [t-59, t]
WIN_2018 = "2018-01-01"
CHUNK = 500_000      # linhas por bloco, para não duplicar a matriz inteira na memória
BASES = ["ret_1", "amplitude", "direcao", "log1p_Volume"]
FAMILY_OF = {"ret_1": "ret_1", "amplitude": "amplitude", "direcao": "direcao",
             "log1p_Volume": "volume"}


def split_positions(feats: pd.DataFrame, window: str) -> dict:
    """Posições (slices) de tr, val e te em feats, usando split() do EDA e split_train_val() do data.py."""
    X_train, y_train, X_test, _ = split(feats)
    x_tr, _, x_val, _ = split_train_val(X_train, y_train)
    if window == "2018":
        x_tr = x_tr[x_tr.index >= WIN_2018]
    elif window != "full":
        raise ValueError(window)
    out = {}
    for part, idx in (("tr", x_tr.index), ("val", x_val.index), ("te", X_test.index)):
        a = feats.index.get_loc(idx[0])
        b = feats.index.get_loc(idx[-1]) + 1
        assert b - a == len(idx), "partes do split deveriam ser contíguas"
        out[part] = slice(a, b)
    return out


def lag_block(base: np.ndarray, rv: np.ndarray, sl: slice, offset: int) -> np.ndarray:
    """Matriz F65 crua das linhas sl de feats. base é (n, 4) com as linhas de feats precedidas
    por `offset` linhas de passado (offset >= N_LAGS), para os lags das primeiras linhas."""
    win = sliding_window_view(base, N_LAGS + 1, axis=0)  # (n-15, 4, 16): win[i, f, k] = base[i+k, f]
    n = sl.stop - sl.start
    X = np.empty((n, 4 * (N_LAGS + 1) + 1), dtype=np.float32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        # linha t de feats está em base[t + offset]; janela que termina nela começa em t+offset-15
        i0 = sl.start + a + offset - N_LAGS
        blk = win[i0:i0 + (b - a), :, ::-1]  # inverte: posição j = lag j
        X[a:b, :-1] = blk.reshape(b - a, -1)
    X[:, -1] = rv[sl]
    return X


def f65_names() -> list:
    return [f"{f}_lag{j}" for f in BASES for j in range(N_LAGS + 1)] + ["rv_60"]


def families_of(names: list) -> dict:
    fam = {}
    for i, name in enumerate(names):
        key = "rv_60" if name == "rv_60" else FAMILY_OF[name.split("_lag")[0]]
        fam.setdefault(key, []).append(i)
    return fam


def scale_inplace(X_tr, others):
    """StandardScaler: fit só em X_tr (em blocos, partial_fit), transform em todos, no lugar."""
    scaler = StandardScaler()
    for a in range(0, len(X_tr), CHUNK):
        scaler.partial_fit(X_tr[a:a + CHUNK])
    for X in [X_tr, *others]:
        for a in range(0, len(X), CHUNK):
            X[a:a + CHUNK] = scaler.transform(X[a:a + CHUNK])
    return scaler.mean_, scaler.scale_


def build_arrays(feats: pd.DataFrame, past: np.ndarray, feature_set: str, window: str) -> dict:
    pos = split_positions(feats, window)
    out = {}
    if feature_set == "F4":
        pre = make_preprocess()
        out["X_tr"] = pre.fit_transform(feats.iloc[pos["tr"]][FEATURES]).astype(np.float32)  # fit só no tr
        out["X_val"] = pre.transform(feats.iloc[pos["val"]][FEATURES]).astype(np.float32)
        out["X_te"] = pre.transform(feats.iloc[pos["te"]][FEATURES]).astype(np.float32)
        preco = pre.named_transformers_["preco"]
        vol = pre.named_transformers_["volume"].named_steps["scale"]
        x_mean = np.r_[preco.mean_, vol.mean_]
        x_scale = np.r_[preco.scale_, vol.scale_]
        names = list(BASES)
    elif feature_set == "F65":
        # base = as 15 linhas de passado (descartadas da 1ª parte) + as linhas de feats
        base = np.concatenate([past, feats[BASES].to_numpy(np.float32)])
        rv = feats["rv_60"].to_numpy(np.float32)
        for part in ("tr", "val", "te"):
            out[f"X_{part}"] = lag_block(base, rv, pos[part], offset=N_LAGS)
        del base
        x_mean, x_scale = scale_inplace(out["X_tr"], [out["X_val"], out["X_te"]])
        names = f65_names()
    else:
        raise ValueError(feature_set)

    for part in ("tr", "val", "te"):
        rows = feats.iloc[pos[part]]
        out[f"y_{part}"] = rows[TARGET].to_numpy(np.float32)
        out[f"ts_{part}"] = rows.index.as_unit("s").asi8.astype(np.int64)  # unix s
        out[f"vol0_{part}"] = (rows["Volume"].to_numpy() == 0)
        out[f"rv60_{part}"] = rows["rv_60"].to_numpy(np.float32)
        out[f"ret1_{part}"] = rows["ret_1"].to_numpy(np.float32)
    out["names"] = names
    out["families"] = families_of(names)
    out["y_std"] = float(feats.iloc[pos["tr"]][TARGET].std())
    out["x_mean"] = np.asarray(x_mean, dtype=np.float64)
    out["x_scale"] = np.asarray(x_scale, dtype=np.float64)
    return out


def load_feats():
    """Lê o CSV (data.py), aplica build_features do EDA e acrescenta log1p(Volume) e rv_60.

    build_features já tira a 1ª linha (ret_1) e a última (alvo). rv_60 = rolling(60) sem
    center: usa ret_1 de t-59 até t (Aula 2, Leakage §3). Descarta as linhas iniciais em que
    rv_60 ou algum lag não existe; devolve também as 15 linhas anteriores (para os lags).
    """
    full = build_features(load_raw())
    full["log1p_Volume"] = np.log1p(full["Volume"])
    full["rv_60"] = full["ret_1"].rolling(RV_WINDOW).std()  # ddof=1, só para trás
    n_drop = max(RV_WINDOW - 1, N_LAGS)
    assert full["rv_60"].iloc[n_drop:].notna().all()
    feats = full.iloc[n_drop:].copy()
    past = full[BASES].iloc[n_drop - N_LAGS:n_drop].to_numpy(np.float32)
    return feats, past


def _save(path: Path, d: dict):
    arrays = {k: v for k, v in d.items() if isinstance(v, np.ndarray)}
    arrays["names"] = np.array(d["names"])
    arrays["families_json"] = np.array(json.dumps(d["families"]))
    arrays["y_std"] = np.array(d["y_std"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["names"] = [str(s) for s in d["names"]]
    d["families"] = json.loads(str(d.pop("families_json")))
    d["y_std"] = float(d["y_std"])
    return d


def load_arrays(feature_set: str = "F4", window: str = "full", _feats=None) -> dict:
    """Interface do contrato. Lê CACHE/{feature_set}_{window}.npz; se não existir, monta e grava."""
    path = CACHE / f"{feature_set}_{window}.npz"
    if path.exists():
        return _load(path)
    feats, past = _feats if _feats is not None else load_feats()
    d = build_arrays(feats, past, feature_set, window)
    _save(path, d)
    return d


if __name__ == "__main__":
    t0 = time.time()
    feats, past = load_feats()
    print(f"CSV lido e base montada em {time.time() - t0:.1f} s ({len(feats)} linhas)")
    for fs in ("F4", "F65"):
        for window in ("full", "2018"):
            t1 = time.time()
            path = CACHE / f"{fs}_{window}.npz"
            if path.exists():
                path.unlink()  # python features.py sempre regera
            d = load_arrays(fs, window, _feats=(feats, past))
            print(f"{fs}_{window}: tr {len(d['y_tr'])} | val {len(d['y_val'])} | te {len(d['y_te'])} | "
                  f"{d['X_tr'].shape[1]} colunas | {time.time() - t1:.1f} s | "
                  f"{path.stat().st_size / 2**20:.0f} MiB")
            del d
    print(f"total {time.time() - t0:.1f} s")
