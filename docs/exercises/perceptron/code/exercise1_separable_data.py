"""Exercise 1 — Separable Data: the case the perceptron was designed for.

Item A: gera as duas classes gaussianas do enunciado (1000 pontos cada) e salva
a Figura 1 em ``figures/``.

Os dados vivem num ``DataFrame`` (colunas ``x1``, ``x2``, ``classe``); a
conversão para matrizes numpy acontece num único ponto, ``as_arrays``, que é por
onde o perceptron do item B vai consumir os dados.

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/exercise1_separable_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

# Parâmetros do enunciado: média e matriz de covariância de cada classe.
CLASSES = {
    0: {"mean": [1.5, 1.5], "cov": [[0.5, 0.0], [0.0, 0.5]]},
    1: {"mean": [5.0, 5.0], "cov": [[0.5, 0.0], [0.0, 0.5]]},
}
N_PER_CLASS = 1000
COLORS = {0: "#1f77b4", 1: "#ff7f0e"}

learning_rate = 0.01


def generate():
    blocos = [
        pd.DataFrame(
            RNG.multivariate_normal(params["mean"], params["cov"], size=N_PER_CLASS),
            columns=["x1", "x2"],
        ).assign(classe=rotulo)
        for rotulo, params in sorted(CLASSES.items())
    ]
    return pd.concat(blocos, ignore_index=True)


def as_arrays(df: pd.DataFrame):
    return df[["x1", "x2"]].to_numpy(), df["classe"].to_numpy()

def predict(x,w,b):
    z = w @ x + b
    if z >= 0:
        return 1
    else:
        return 0

def update(x,y,w,b):
    y_pred = predict(x,w,b)
    novo_w = w + learning_rate * (y-y_pred) * x
    novo_b = b + learning_rate * (y-y_pred)
    return novo_w, novo_b

def accuracy(X, y, w, b):
    return float(np.mean([predict(xi, w, b) for xi in X] == y))

def inicialize():
    w = RNG.normal(0,0.01,size=2)
    b = 0.0
    return w, b

def train(X, y, max_epocas=100):
    """Treina até uma passada inteira não produzir atualização, ou 100 épocas."""
    historico = []
    w, b = inicialize()
    for epoca in range(1, max_epocas + 1):
        atualizacoes = 0
        for xi, yi in zip(X, y):
            erro = yi - predict(xi, w, b)
            if erro != 0:                    
                w, b = update(xi,yi,w,b)
                atualizacoes += 1

        historico.append({"epoca": epoca,
                          "atualizacoes": atualizacoes,
                          "acuracia": accuracy(X, y, w, b)})
        if atualizacoes == 0:
            break                            # convergiu

    return w, b, pd.DataFrame(historico)


def figure1(df: pd.DataFrame) -> None:
    """Figura 1 — dispersão das 2 classes, uma cor por classe."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for rotulo, grupo in df.groupby("classe"):
        ax.scatter(grupo["x1"], grupo["x2"], s=14, alpha=0.7,
                   color=COLORS[rotulo], label=f"Classe {rotulo}")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 — Duas classes separáveis")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-separable-data.png", dpi=150)
    plt.close(fig)  # (2)!


def figure2(df: pd.DataFrame, w, b) -> None:
    """Figura 2 — fronteira aprendida sobre os dados, com os erros destacados."""
    fig, ax = plt.subplots(figsize=(8, 6))

    X, y = as_arrays(df)
    pred = np.array([predict(xi, w, b) for xi in X])
    erros = pred != y

    for rotulo, grupo in df[~erros].groupby("classe"):
        ax.scatter(grupo["x1"], grupo["x2"], s=14, alpha=0.7,
                   color=COLORS[rotulo], label=f"Classe {rotulo}")

    if erros.any():
        ax.scatter(df.loc[erros, "x1"], df.loc[erros, "x2"], s=70,
                   facecolors="none", edgecolors="red", linewidths=1.6,
                   zorder=4, label=f"Mal classificados ({erros.sum()})")

    xs = np.array([df["x1"].min() - 0.5, df["x1"].max() + 0.5])
    ax.plot(xs, -(w[0] * xs + b) / w[1], color="black", lw=2, zorder=5,
            label=r"Fronteira $\mathbf{w}\cdot\mathbf{x} + b = 0$")

    ax.set_xlim(df["x1"].min() - 0.5, df["x1"].max() + 0.5)
    ax.set_ylim(df["x2"].min() - 0.5, df["x2"].max() + 0.5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Figura 2 — Fronteira aprendida")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-decision-boundary.png", dpi=150)
    plt.close(fig)


def figure3(df: pd.DataFrame) -> None:
    """Figura 3 — acurácia x época."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["epoca"], df["acuracia"], marker="o", ms=3)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Figura 3 — Acurácia por época")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-accuracy-per-epoch.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = generate()
    figure1(df)
    X, y = as_arrays(df)
    w, b, historico = train(X, y)
    figure2(df, w, b)
    figure3(historico)
    print(f"w final = [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"b final ={b:.4f}")
    print(f"épocas = {historico['epoca'].iloc[-1]}")
    print(f"acurácia = {historico['acuracia'].iloc[-1]:.2%}")
    
    



if __name__ == "__main__":
    main()
