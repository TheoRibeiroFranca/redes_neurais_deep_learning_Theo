
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

    df[TARGET].plot.hist(bins=100, title="Histograma de Close")
    plt.xlabel("Preço de fechamento (USD)")
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




def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("Timestamp")
    describe(df)

    print(f"\nTipos de feature\n  numéricas  : {COLUMNS}")
    


if __name__ == "__main__":
    main()
