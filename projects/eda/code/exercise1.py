
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "data" / "btcusd_1-min_data.csv"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
SEED = 42  # mesma semente do resto do relatório

COLUMNS = ["Open", "High", "Low", "Volume"]
TARGET = "Close"


def describe(df: pd.DataFrame):
    print(f"Forma do arquivo bruto\n  {df.shape}\n")

    close_decribe = df[TARGET].describe()
    close_assimetria = df[TARGET].skew()
    print("estatísticas de Close")
    print(close_decribe)
    print(close_assimetria)

    # Figura 1: Close e log(Close) lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(df[TARGET], bins=100)
    ax1.set_title("Close (USD)")
    ax2.hist(np.log(df[TARGET]), bins=100)
    ax2.set_title("log(Close)")
    plt.savefig(FIGURES / "close_hist.png", dpi=120, bbox_inches="tight")
    plt.close()


    print("\nValores ausentes por coluna (markdown)")
    missing = pd.DataFrame({
        "Ausentes": df.isna().sum(),
        "%": (df.isna().mean() * 100).round(2),
    })
    missing = missing[missing["Ausentes"] > 0].sort_values("Ausentes", ascending=False)
    print("| Coluna | Ausentes | % do total |")
    for col, row in missing.iterrows():
        print(f"| `{col}` | {int(row['Ausentes'])} | {row['%']:.2f}% |")
    print(f"  total de células ausentes: {int(df.isna().sum().sum())}")

    print("\nColunas de variáveis independentes: média, mediana e máximo (markdown)")
    print("| Coluna | Média | Mediana | Máximo | % de zeros |")
    for col in COLUMNS:
        mean, median, top = df[col].mean(), df[col].median(), df[col].max()
        zeros = df[col].eq(0).mean() * 100
        print(f"| `{col}` | {mean:.2f} | {median:.2f} | {top:.0f} | {zeros:.1f}% |")

    print("\nContinuidade da série")
    print(f"  período: {df.index.min()} → {df.index.max()}")
    gaps = df.index.to_series().diff().value_counts()
    print("  intervalos entre linhas consecutivas (intervalo → quantidade):")
    for gap, count in gaps.items():
        print(f"    {gap} → {count}")
    print(f"  timestamps repetidos: {df.index.duplicated().sum()}")

    print(f"  linhas duplicadas   : {df.reset_index().duplicated().sum()}")


def quality(df: pd.DataFrame):
    print("\nRegras do candle (linhas que violam)")
    print("preço <= 0:", (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
    print("Volume < 0:", (df["Volume"] < 0).sum())
    print("High < Low:", (df["High"] < df["Low"]).sum())
    print("Open fora:", ((df["Open"] < df["Low"]) | (df["Open"] > df["High"])).sum())
    print("Close fora:", ((df["Close"] < df["Low"]) | (df["Close"] > df["High"])).sum())

    print("\nMinutos sem negócio")
    sem_negocio = df["Volume"] == 0
    precos_iguais = (df["Open"] == df["Close"]) & (df["High"] == df["Close"]) & (df["Low"] == df["Close"])
    print("sem negócio:", sem_negocio.sum())
    print("sem negócio e preços iguais:", (sem_negocio & precos_iguais).sum())

    print("\nCorrelação com Close")
    print(df.corr()[TARGET])


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("Timestamp")
    describe(df)

    print(f"\nTipos de feature\n  numéricas  : {COLUMNS}")

    quality(df)

    cut = '2024-01-01'
    train, test = df[df.index < cut], df[df.index >= cut]

    print("\nSplit temporal")
    for nome, parte in [("treino", train), ("teste", test)]:
        print(nome, len(parte), len(parte) / len(df) * 100, parte.index.min(), parte.index.max())
        print("  Close mín/máx/média:", parte[TARGET].min(), parte[TARGET].max(), parte[TARGET].mean())
    print("% teste acima do máx do treino:", (test[TARGET] > train[TARGET].max()).mean() * 100)

    # Figura 2: Close diário, treino e teste em cores diferentes
    plt.figure(figsize=(11, 4))
    plt.plot(train[TARGET].resample("D").last(), label="treino")
    plt.plot(test[TARGET].resample("D").last(), label="teste")
    plt.title("Close diário e o corte do split")
    plt.ylabel("USD")
    plt.legend()
    plt.savefig(FIGURES / "close_split.png", dpi=120, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()
