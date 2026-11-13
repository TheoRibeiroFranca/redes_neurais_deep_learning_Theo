# Estatísticas do alvo ret_futuro: o tr é a base; a val só compara o regime; o teste não entra.
# Aula 2 · Distributions: forma da coluna (assimetria, curtose, zeros) antes de escolher a perda.
# Aula 8.2 · Regression: alvo com cauda pesada; quantis de |alvo| além de média e desvio.
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

from data import load_raw, split_train_val
from exercise4 import CSV, TARGET, build_features, split

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"


def estatisticas(y: np.ndarray) -> dict:
    a = np.abs(y)
    return {
        "n": int(len(y)),
        "media": float(y.mean()),
        "desvio": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "assimetria": float(skew(y)),
        "curtose_excesso": float(kurtosis(y)),  # Fisher: normal = 0
        "abs_quantis": {
            "p50": float(np.quantile(a, 0.50)),
            "p90": float(np.quantile(a, 0.90)),
            "p99": float(np.quantile(a, 0.99)),
            "p99_9": float(np.quantile(a, 0.999)),
        },
        "pct_zero": float(100 * np.mean(y == 0)),
    }


def correlacoes(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "spearman": float(spearmanr(x, y)[0]),
        "pearson": float(pearsonr(x, y)[0]),
    }


def particao(index, n: int) -> dict:
    return {"inicio": str(index.min()), "fim": str(index.max()), "n": int(n)}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    feats = build_features(raw)
    X_train, y_train, _X_test, _y_test = split(feats)  # o teste vem do split e não é usado
    x_tr, y_tr, x_val, y_val = split_train_val(X_train, y_train)

    ret_tr = x_tr["ret_1"].to_numpy(dtype=np.float64)
    ret_val = x_val["ret_1"].to_numpy(dtype=np.float64)
    y_tr_a = y_tr.to_numpy(dtype=np.float64)
    y_val_a = y_val.to_numpy(dtype=np.float64)

    # Volume == 0 por ano: informativo, sobre TODAS as linhas do CSV (inclui validação e teste).
    vol0 = (raw["Volume"] == 0)
    anos = {}
    for ano, grupo in vol0.groupby(raw.index.year):
        anos[str(ano)] = {"n": int(len(grupo)), "pct_volume_zero": float(100 * grupo.mean())}

    out = {
        "versao_csv": {
            "arquivo": str(CSV.name),
            "linhas": int(len(raw)),
            "inicio": str(raw.index.min()),
            "fim": str(raw.index.max()),
        },
        "particoes": {
            "tr": particao(x_tr.index, len(y_tr_a)),
            "val": particao(x_val.index, len(y_val_a)),
            "obs": "tr e val saem do split oficial de data.py (split + split_train_val); teste nao usado",
        },
        "alvo": TARGET,
        "tr": estatisticas(y_tr_a),
        "val": estatisticas(y_val_a),
        "correlacao_ret1_alvo": {
            "tr": correlacoes(ret_tr, y_tr_a),
            "val": correlacoes(ret_val, y_val_a),
        },
        "volume_zero_por_ano": {
            "escopo": "ARQUIVO INTEIRO (todas as linhas do CSV, inclui validacao e teste). Contagem informativa, nao usada para escolha de modelo.",
            "anos": anos,
        },
    }

    with open(RES / "target_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Figura 1: histograma do alvo no tr, contagem em escala log
    fora = int(np.sum(np.abs(y_tr_a) > 0.02))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(y_tr_a.min(), y_tr_a.max(), 401)
    ax.hist(y_tr_a, bins=bins, color="tab:blue", label="minutos do treino (2012-2022)")
    ax.axvline(0.0, color="black", linewidth=0.8, label="alvo = 0")
    ax.set_yscale("log")
    ax.set_title("Alvo ret_futuro no treino: cauda pesada")
    ax.set_xlabel("retorno logarítmico do minuto seguinte, log(Close[t+1] / Close[t])")
    ax.set_ylabel("contagem de minutos (escala log)")
    ax.text(0.98, 0.95, f"{fora} minutos com |alvo| > 0,02", transform=ax.transAxes,
            ha="right", va="top", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_alvo_hist.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: out[k] for k in ("versao_csv", "particoes")}, indent=2, ensure_ascii=False))
    for parte in ("tr", "val"):
        print(parte, json.dumps(out[parte], indent=2, ensure_ascii=False))
    print("correlacoes", json.dumps(out["correlacao_ret1_alvo"], indent=2))
    print("volume zero por ano", {a: round(v["pct_volume_zero"], 2) for a, v in anos.items()})


if __name__ == "__main__":
    main()
