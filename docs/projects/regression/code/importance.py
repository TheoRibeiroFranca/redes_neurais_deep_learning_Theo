# Importância por permutação por família de colunas, na validação. NÃO é de aula:
#   sklearn.inspection.permutation_importance (documentação do scikit-learn) e
#   Breiman, L. (2001), Random Forests, Machine Learning 45, 5-32. Exigida pela rubrica.
# Aula 2 (Splitting, "The golden rule"): usa só a validação; o teste não é lido aqui.
# Aula 2 (Scaling the target): a saída da rede volta à escala original (x y_std) antes de medir.
# Aula 8.2 (Regression): a métrica é RMSE na escala original.
# Aula 7 (Forgetting model.eval): model.eval() e no_grad na inferência.
# O Ridge do mesmo conjunto entra como comparação. Mesma sequência de permutações para os dois.
import csv
import json
import os
from pathlib import Path

import pickle

import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error

from features import load_arrays
from model import MLP


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RES = ROOT / "docs" / "projects" / "regression" / "results"

SEED = 42
REPETICOES = 5
BATCH = 65536


def carregar_rede(cfg: dict, n_in: int, nome: str) -> MLP:
    rede = MLP(n_in, list(cfg["hidden"]), dropout=float(cfg["dropout"]))
    estado = torch.load(RES / "models" / f"{nome}.pt", map_location="cpu", weights_only=True)
    rede.load_state_dict(estado)
    rede.eval()
    return rede


@torch.no_grad()
def prever_rede(rede: MLP, X: np.ndarray, y_std: float) -> np.ndarray:
    saida = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), BATCH):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH], dtype=np.float32))
        saida[i:i + BATCH] = rede(xb).squeeze(-1).numpy()
    return saida * y_std


def importancia(nome: str, prever, X: np.ndarray, y: np.ndarray, familias: dict) -> list:
    base = root_mean_squared_error(y, prever(X))
    linhas = []
    for familia, cols in familias.items():
        cols = np.asarray(cols)
        rng = np.random.default_rng(SEED)  # reiniciado por família: permutações pareadas entre modelos
        Xp = X.copy()
        deltas = []
        for _ in range(REPETICOES):
            perm = rng.permutation(len(X))
            # embaralha as colunas da família juntas: a mesma permutação de linhas em todas elas
            Xp[:, cols] = X[np.ix_(perm, cols)]
            deltas.append(root_mean_squared_error(y, prever(Xp)) - base)
        linhas.append({
            "modelo": nome,
            "familia": familia,
            "n_colunas": int(len(cols)),
            "rmse_base": float(base),
            "delta_rmse_media": float(np.mean(deltas)),
            "delta_rmse_dp": float(np.std(deltas, ddof=1)),
            "repeticoes": REPETICOES,
            "seed": SEED,
        })
        print(f"{nome} | {familia}: dRMSE {np.mean(deltas):+.8f} +- {np.std(deltas, ddof=1):.8f}")
    return linhas


def main() -> None:
    torch.set_num_threads(int(os.environ.get("NUM_THREADS", 3)))
    chosen = json.loads((RES / "chosen.json").read_text(encoding="utf-8"))
    cfg = chosen["config"]
    fs, window = cfg["feature_set"], cfg["window"]

    arr = load_arrays(fs, window)
    X, y = arr["X_val"], arr["y_val"]
    y_std = float(arr["y_std"])
    familias = arr["families"]
    del arr

    rede = carregar_rede(cfg, X.shape[1], "final_s42")
    with open(RES / "models" / f"ridge_{fs}_{window}.pkl", "rb") as f:
        ridge = pickle.load(f)

    linhas = importancia("rede_final_s42", lambda A: prever_rede(rede, A, y_std), X, y, familias)
    linhas += importancia("ridge_mesmo_conjunto", ridge.predict, X, y, familias)

    with open(RES / "importance_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"Gravado: {RES / 'importance_val.csv'}")


if __name__ == "__main__":
    main()
