"""Exercise 2 — Overlapping Data: the case the perceptron cannot solve.

Reusa a implementação do Exercício 1 sem alterações (mesmo eta = 0.01, mesmo
teto de 100 épocas) e acrescenta apenas o *pocket algorithm*: sempre que uma
atualização produz acurácia melhor que qualquer uma já vista, o par (w, b) é
copiado para o "bolso".

Salva as Figuras 4, 5 e 6 em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/exercise2_non_separable_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

# Parâmetros do enunciado: média e matriz de covariância de cada classe.
CLASSES = {
    0: {"mean": [3.0, 3.0], "cov": [[1.5, 0.0], [0.0, 1.5]]},
    1: {"mean": [4.0, 4.0], "cov": [[1.5, 0.0], [0.0, 1.5]]},
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
    """Treina com o pocket algorithm: guarda o melhor (w, b) já visto.

    Os dados não são separáveis, então ``atualizacoes == 0`` nunca acontece e o
    laço roda as 100 épocas inteiras. O pocket é checado a cada atualização,
    como pede o enunciado — são poucas centenas no total, custo irrelevante.
    """
    historico = []
    w, b = inicialize()
    melhor_w, melhor_b = w.copy(), b
    melhor_acuracia = accuracy(X, y, w, b)

    for epoca in range(1, max_epocas + 1):
        atualizacoes = 0
        for xi, yi in zip(X, y):
            erro = yi - predict(xi, w, b)
            if erro != 0:
                w, b = update(xi, yi, w, b)
                atualizacoes += 1

                acc = accuracy(X, y, w, b)
                if acc > melhor_acuracia:
                    melhor_acuracia, melhor_w, melhor_b = acc, w.copy(), b

        historico.append({"epoca": epoca,
                          "atualizacoes": atualizacoes,
                          "acuracia": accuracy(X, y, w, b),
                          "melhor_acuracia": melhor_acuracia})
        if atualizacoes == 0:
            break

    return w, b, melhor_w, melhor_b, pd.DataFrame(historico)


def figure4(df: pd.DataFrame) -> None:
    """Figura 4 — dispersão das 2 classes, uma cor por classe."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for rotulo, grupo in df.groupby("classe"):
        ax.scatter(grupo["x1"], grupo["x2"], s=14, alpha=0.7,
                   color=COLORS[rotulo], label=f"Classe {rotulo}")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 4 — Duas classes não separáveis")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-non-separable-data.png", dpi=150)
    plt.close(fig)  # (2)!


def figure5(df: pd.DataFrame, w, b, w_melhor, b_melhor) -> None:
    """Figura 5 — fronteira aprendida sobre os dados, com os erros destacados."""
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
            label=r"Ultima fronteira")
    ax.plot(xs, -(w_melhor[0] * xs + b_melhor) / w_melhor[1], color="blue", lw=2, zorder=5,
                label=r"Melhor fronteira")
    

    ax.set_xlim(df["x1"].min() - 0.5, df["x1"].max() + 0.5)
    ax.set_ylim(df["x2"].min() - 0.5, df["x2"].max() + 0.5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Figura 5 — Fronteira aprendida")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-decision-boundary-non-seperable-data.png", dpi=150)
    plt.close(fig)


def figure6(df: pd.DataFrame) -> None:
    """Figura 6 — acurácia atual e melhor acuracia x época."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["epoca"], df["acuracia"], marker="o", ms=3, label="acuracia real")
    ax.plot(df["epoca"], df["melhor_acuracia"], marker="s", ms=3, label="melhor acuracia")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Figura 6 — Acurácia por época")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-accuracy-per-epoch-non-seperable-data.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = generate()
    figure4(df)
    X, y = as_arrays(df)
    w, b, melhor_w, melhor_b, historico = train(X, y)
    figure5(df, w, b, melhor_w, melhor_b)
    figure6(historico)
    print(f"w final = [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"b final ={b:.4f}")
    print(f"melhor w  = [{melhor_w[0]:.4f}, {melhor_w[1]:.4f}]")
    print(f"melhor b  ={melhor_b:.4f}")
    print(f"épocas = {historico['epoca'].iloc[-1]}")
    print(f"acurácia = {historico['acuracia'].iloc[-1]:.2%}")
    print(f"melhor acurácia = {historico['melhor_acuracia'].iloc[-1]:.2%}")

    



if __name__ == "__main__":
    main()
