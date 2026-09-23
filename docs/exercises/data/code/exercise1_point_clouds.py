"""Exercise 1 — Point Clouds: Geometry and Spread in 2D.

Gera as 4 classes gaussianas do enunciado em quatro escalas de dispersão,
salva as Figuras 1, 1b, 2 e 3 em ``figures/`` e imprime todos os números
citados no relatório.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

# Parâmetros do enunciado: média e desvio-padrão por eixo, para cada classe.
CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}
N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)

# Médias empilhadas (4, 2) — não dependem de s, servem aos dois indicadores.
MEANS = np.array([CLASSES[c]["mean"] for c in sorted(CLASSES)])
STDS = np.array([CLASSES[c]["std"] for c in sorted(CLASSES)])
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def generate(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Amostra 100 pontos por classe com os desvios multiplicados por ``scale``.

    As médias nunca mudam: ``scale`` mexe só na dispersão, que é exatamente o
    experimento pedido no item B.
    """
    xs, ys = [], []
    for label in sorted(CLASSES):
        mean = MEANS[label]
        std = STDS[label] * scale
        xs.append(RNG.normal(mean, std, size=(N_PER_CLASS, 2)))
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def separation_ratios() -> dict[tuple[int, int], float]:
    """r_ij = ||mu_i - mu_j|| / (sigma_i_barra + sigma_j_barra), para s = 1.

    Usa os parâmetros do enunciado (não as estimativas da amostra), como pede a
    fórmula: sigma_k_barra é a média dos dois desvios-padrão da classe k.
    """
    sigma_bar = STDS.mean(axis=1)  # (sigma_x + sigma_y) / 2 por classe
    return {
        (i, j): float(np.linalg.norm(MEANS[i] - MEANS[j]) / (sigma_bar[i] + sigma_bar[j]))
        for i, j in combinations(sorted(CLASSES), 2)
    }


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fração de pontos cujo centro de classe mais próximo não é o da própria classe.

    Medida puramente geométrica: distância de cada ponto às 4 médias teóricas,
    via broadcasting (N, 1, 2) - (1, 4, 2) -> (N, 4). Nada é treinado.
    """
    distances = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    return float((nearest != y).mean())


def _scatter(ax: plt.Axes, X: np.ndarray, y: np.ndarray, *, centers: bool) -> None:
    """Desenha as 4 nuvens em ``ax``, opcionalmente marcando os centros."""
    for label in sorted(CLASSES):
        ax.scatter(*X[y == label].T, s=14, alpha=0.7, color=COLORS[label],
                   label=f"Classe {label}")
    if centers:
        ax.scatter(*MEANS.T, marker="X", s=180, c="black", zorder=5,
                   label="Centros $\\mu_k$")
        for label in sorted(CLASSES):
            ax.annotate(f"$\\mu_{label}$", MEANS[label], textcoords="offset points",
                        xytext=(8, 8), fontsize=11, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def figure1(X: np.ndarray, y: np.ndarray) -> None:
    """Figura 1 — dispersão das 4 classes com os centros marcados (item A)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, X, y, centers=True)
    ax.set_title("Figura 1 — Nuvens de pontos das 4 classes ($s = 1.0$)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)  # (2)!


def figure1b(X: np.ndarray, y: np.ndarray) -> None:
    """Figura 1b — as mesmas nuvens com as fronteiras esboçadas (item C).

    O esboço é a partição de Voronoi das 4 médias: o que uma rede treinada
    aprenderia se os custos fossem simétricos e as dispersões parecidas. Fica
    calculado em vez de desenhado à mão, para não inventar fronteira.
    """
    pad = 2.0
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 600)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 600)
    grid = np.stack(np.meshgrid(gx, gy), axis=-1)          # (600, 600, 2)
    d = np.linalg.norm(grid[:, :, None, :] - MEANS[None, None, :, :], axis=3)
    regions = d.argmin(axis=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(gx, gy, regions, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=COLORS, alpha=0.15)
    ax.contour(gx, gy, regions, levels=[0.5, 1.5, 2.5], colors="black",
               linewidths=1.2, linestyles="--")
    _scatter(ax, X, y, centers=True)
    ax.set_title("Figura 1b — Fronteiras de decisão esboçadas ($s = 1.0$)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01b-decision-boundaries.png", dpi=150)
    plt.close(fig)


def figure2(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    """Figura 2 — 4 subplots (um por s) com os MESMOS limites de eixo.

    Limites compartilhados são o que torna a comparação honesta: sem isso o
    matplotlib reescala cada painel e as quatro nuvens parecem idênticas.
    """
    all_points = np.vstack([X for X, _ in datasets.values()])
    pad = 1.0
    xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, scale in zip(axes.ravel(), SCALES):
        X, y = datasets[scale]
        _scatter(ax, X, y, centers=True)
        ax.set_title(f"$s = {scale}$  —  taxa de mistura = {mixing_rate(X, y):.1%}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.suptitle("Figura 2 — Mesmas 4 classes sob quatro fatores de dispersão "
                 "(eixos compartilhados)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-scale-grid.png", dpi=150)
    plt.close(fig)


def figure3(rates: dict[float, float]) -> None:
    """Figura 3 — taxa de mistura × fator de escala s."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SCALES, [rates[s] for s in SCALES], marker="o", color="#d62728",
            label="Taxa de mistura")
    for scale in SCALES:
        ax.annotate(f"{rates[scale]:.1%}", (scale, rates[scale]),
                    textcoords="offset points", xytext=(6, 8))
    ax.set_xlabel("Fator de escala $s$ (multiplicador dos desvios-padrão)")
    ax.set_ylabel("Fração de pontos com centro mais próximo errado")
    ax.set_title("Figura 3 — Taxa de mistura em função de $s$")
    ax.set_xticks(SCALES)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Gera os 4 datasets de uma vez só, em ordem fixa, e reusa em todas as
    # figuras e métricas: assim a mesma amostra de s = 1 aparece na Figura 1,
    # na Figura 2 e nas contas.
    datasets = {scale: generate(scale) for scale in SCALES}

    figure1(*datasets[1.0])
    figure1b(*datasets[1.0])
    figure2(datasets)

    rates = {scale: mixing_rate(*datasets[scale]) for scale in SCALES}
    figure3(rates)

    print("== Taxa de mistura por fator de escala ==")
    for scale in SCALES:
        print(f"  s = {scale:>4} | taxa de mistura = {rates[scale]:.4f}  ({rates[scale]:.2%})")

    print("\n== Separation ratio r_ij (s = 1.0, parâmetros do enunciado) ==")
    ratios = separation_ratios()
    for (i, j), r in sorted(ratios.items()):
        print(f"  r_{i}{j} = {r:.3f}")
    pair, smallest = min(ratios.items(), key=lambda kv: kv[1])
    print(f"  menor: r_{pair[0]}{pair[1]} = {smallest:.3f}")
    # r_ij é inversamente proporcional a s: as médias não mudam e o denominador
    # inteiro é multiplicado por s. Logo r_ij(s) = r_ij(1) / s.
    print(f"  o mesmo par em s = 2.0: {smallest:.3f} / 2 = {smallest / 2:.3f}")


if __name__ == "__main__":
    main()
