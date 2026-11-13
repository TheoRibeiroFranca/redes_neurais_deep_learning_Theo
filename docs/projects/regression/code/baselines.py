# Baselines da entrega 2 (alvo ret_futuro), só na validação de 2023. O teste não é lido.
# Aula 8.2 (Regression), "Always compute the baseline": média, mediana e último valor observado (persistência).
# Aula 2 (Leakage §4, Pipeline): fit só no treino; a validação só é usada para medir e para escolher o alpha.
# Aula 7 (Regularization): penalização L2 (Ridge) e aviso sobre a escolha pela validação.
# Métrica primária: RMSE (contrato). Ridge: solução fechada, em blocos, em float64 (ver ridge_fit_blocks).

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

NUM_THREADS = os.environ.get("NUM_THREADS", "3")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, NUM_THREADS)

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from features import load_arrays  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "docs" / "projects" / "regression" / "results"
CACHE = ROOT / "data" / "cache"

SEED = 42  # não há aleatoriedade aqui (solução fechada); fixado por convenção do projeto
FEATURE_SETS = ["F4", "F65"]
WINDOWS = ["full", "2018"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
BLOCK = 1_000_000  # linhas por bloco ao acumular X'X (evita copiar 5,8 M x 65 em float64)


def metrics(y, yhat):
    return {
        "rmse": float(root_mean_squared_error(y, yhat)),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
    }


def ridge_fit_blocks(X, y, alpha):
    """Ridge com intercepto não penalizado (mesma definição do sklearn), em dois passes por blocos.

    Passe 1: médias de X e y. Passe 2: X'X e X'y centrados. Depois w = (X'X + alpha I)^-1 X'y.
    """
    n, p = X.shape
    mu_x = np.zeros(p)
    mu_y = 0.0
    for i in range(0, n, BLOCK):
        mu_x += X[i : i + BLOCK].astype(np.float64).sum(axis=0)
        mu_y += float(y[i : i + BLOCK].astype(np.float64).sum())
    mu_x /= n
    mu_y /= n

    gram = np.zeros((p, p))
    xty = np.zeros(p)
    for i in range(0, n, BLOCK):
        xb = X[i : i + BLOCK].astype(np.float64) - mu_x
        yb = y[i : i + BLOCK].astype(np.float64) - mu_y
        gram += xb.T @ xb
        xty += xb.T @ yb

    coef = np.linalg.solve(gram + alpha * np.eye(p), xty)
    intercept = mu_y - mu_x @ coef
    return coef, intercept


def ridge_object(alpha, coef, intercept, n_features):
    """Objeto sklearn Ridge com os atributos ajustados preenchidos (para o pkl e o predict padrão)."""
    model = Ridge(alpha=alpha)
    model.coef_ = coef
    model.intercept_ = intercept
    model.n_features_in_ = n_features
    return model


def run(res_dir: Path, cache_dir: Path):
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "models").mkdir(parents=True, exist_ok=True)
    (cache_dir / "preds").mkdir(parents=True, exist_ok=True)

    rows = []
    ridge_best_per_combo = {}
    extras = {}  # (fs, window) -> arrays da validação para salvar as previsões do melhor Ridge

    for fs in FEATURE_SETS:
        for win in WINDOWS:
            d = load_arrays(fs, win)  # só as chaves de treino e de validação são usadas
            X_tr, y_tr = d["X_tr"], d["y_tr"]
            X_val, y_val = d["X_val"], d["y_val"]
            y_val64 = y_val.astype(np.float64)
            n_val = len(y_val)
            print(f"[{fs}/{win}] tr={len(y_tr)} val={n_val} p={X_tr.shape[1]}", flush=True)

            # Dummy média e mediana: DummyRegressor ignora X, então passa-se uma coluna de zeros.
            zero_x_tr = np.zeros((len(y_tr), 1), dtype=np.float32)
            zero_x_val = np.zeros((n_val, 1), dtype=np.float32)
            dummy_mean = DummyRegressor(strategy="mean").fit(zero_x_tr, y_tr)
            dummy_median = DummyRegressor(strategy="median").fit(zero_x_tr, y_tr)
            preds = {
                ("dummy_mean", None): dummy_mean.predict(zero_x_val).astype(np.float64),
                ("dummy_median", None): dummy_median.predict(zero_x_val).astype(np.float64),
                ("zero", None): np.zeros(n_val),
                ("persistencia", None): d["ret1_val"].astype(np.float64),  # ret_1 cru do minuto t
            }

            # Ridge: alpha escolhido pelo RMSE na validação. Ajuste só com o treino.
            ridge_models = {}
            for alpha in ALPHAS:
                coef, intercept = ridge_fit_blocks(X_tr, y_tr, alpha)
                model = ridge_object(alpha, coef, intercept, X_tr.shape[1])
                yhat = X_val.astype(np.float64) @ coef + intercept
                preds[("ridge", alpha)] = yhat
                ridge_models[alpha] = model

            for (name, alpha), yhat in preds.items():
                if name == "ridge":
                    continue
                rows.append({"feature_set": fs, "window": win, "model": name, "alpha": np.nan,
                             **metrics(y_val64, yhat)})
            for alpha in ALPHAS:
                m = metrics(y_val64, preds[("ridge", alpha)])
                rows.append({"feature_set": fs, "window": win, "model": "ridge", "alpha": alpha, **m})

            best_alpha = min(ALPHAS, key=lambda a: metrics(y_val64, preds[("ridge", a)])["rmse"])
            ridge_best_per_combo[f"{fs}_{win}"] = {"alpha": best_alpha,
                                                   **metrics(y_val64, preds[("ridge", best_alpha)])}
            with open(res_dir / "models" / f"ridge_{fs}_{win}.pkl", "wb") as f:
                pickle.dump(ridge_models[best_alpha], f)

            extras[(fs, win)] = {
                "y": y_val64,
                "yhat": preds[("ridge", best_alpha)],
                "ts": d["ts_val"],
                "vol0": d["vol0_val"],
                "rv60": d["rv60_val"],
                "ret1": d["ret1_val"],
                "rmse": ridge_best_per_combo[f"{fs}_{win}"]["rmse"],
            }
            del d, X_tr, y_tr, X_val, y_val

    table = pd.DataFrame(rows, columns=["feature_set", "window", "model", "alpha", "rmse", "mae", "r2"])
    table = table.sort_values("rmse").reset_index(drop=True)
    table.to_csv(res_dir / "baselines_val.csv", index=False, float_format="%.8g")

    best_row = table.iloc[0]
    best_json = {
        "criterio": "rmse na validação (2023); menor entre todas as linhas de baselines_val.csv",
        "split": "val",
        "seed": SEED,
        "n_configs_tentadas_ridge": len(ALPHAS) * len(FEATURE_SETS) * len(WINDOWS),
        "melhor": {
            "feature_set": best_row["feature_set"],
            "window": best_row["window"],
            "model": best_row["model"],
            "alpha": None if pd.isna(best_row["alpha"]) else float(best_row["alpha"]),
            "rmse": float(best_row["rmse"]),
            "mae": float(best_row["mae"]),
            "r2": float(best_row["r2"]),
        },
        "ridge_escolhido_por_combo": ridge_best_per_combo,
    }
    with open(res_dir / "baselines_best.json", "w", encoding="utf-8") as f:
        json.dump(best_json, f, indent=2, ensure_ascii=False)

    # Previsões de validação do melhor Ridge (entre as 4 combinações), para o par 4 testar.
    key = min(ridge_best_per_combo, key=lambda k: ridge_best_per_combo[k]["rmse"])
    fs, win = key.split("_", 1)
    e = extras[(fs, win)]
    np.savez_compressed(
        cache_dir / "preds" / "ridge_best_val.npz",
        y=e["y"], yhat=e["yhat"], ts=e["ts"], vol0=e["vol0"], rv60=e["rv60"], ret1=e["ret1"],
        split=np.array("val"),
    )

    print("\nbaselines_val.csv (validação, ordenado por rmse):")
    print(table.to_string(index=False))
    print("\nmelhor pela validação:", json.dumps(best_json["melhor"], ensure_ascii=False))
    print("ridge escolhido por combinação:", json.dumps(ridge_best_per_combo, ensure_ascii=False))
    print(f"ridge_best_val.npz <- Ridge {fs}/{win}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Baselines na validação (sem teste).")
    ap.add_argument("--results", type=Path, default=RES, help="pasta de saída (padrão: results/)")
    ap.add_argument("--cache", type=Path, default=CACHE, help="pasta do cache (padrão: data/cache/)")
    args = ap.parse_args()
    run(args.results, args.cache)
