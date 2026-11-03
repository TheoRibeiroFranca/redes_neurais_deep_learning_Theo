---
project: regression
dataset: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
team:
  - Theo Ribeiro França
  - Lucas Sato Coelho de Oliveira
  - Rafael Kauffmann
ai_use: "estruturação do index.md (TODO: completar com o uso real de IA nesta entrega)"
---

# 2. Regressão

!!! abstract "Entrega 2 de 3 do [Projeto](../index.md)"

    [Projects → Classification or Regression](https://insper.github.io/ann-dl/2026.2/projects/supervised/){:target='_blank'} · prazo **05.nov.2026, 23:59**

!!! info "Equipe"

    | Nome completo | GitHub |
    |---------------|--------|
    | Theo Ribeiro França | TheoRibeiroFranca |
    | Lucas Sato Coelho de Oliveira | lucassatoco |
    | Rafael Kauffmann | RafaelKauffmann |

    Dataset, decisões e status: [página do projeto](../index.md).

Rede neural treinada no mesmo dataset, com a mesma tarefa e o mesmo split do
[EDA](../eda/index.md): prever o **retorno logarítmico do minuto seguinte** do BTC/USD
(`ret_futuro = log(Close[t+1] / Close[t])`) a partir de `ret_1`, `amplitude`, `direcao` e
`log1p(Volume)`. A pergunta que guia a entrega: *a rede aprende algo que modelos simples não
aprendem? E até que ponto dá para confiar no número que ela reporta?*

## 0. Ponto de partida

O pipeline do EDA é importado de [`../eda/code/exercise4.py`](../eda/code/exercise4.py)
(`build_features`, `split`, `make_preprocess`), sem cópia, e ajustado **só no treino**.

| | Treino | Validação | Teste |
|---|--------|-----------|-------|
| Período | 2012-01-01 → TODO | TODO → 2023-12-31 | 2024-01-01 → 2026-10-06 |
| Linhas | TODO | TODO | 1 453 183 |

A validação sai do **fim do treino, por corte temporal**, como o EDA previu
([EDA §9](../eda/index.md#9-estrategia-de-split)). O teste é usado **uma única vez**, na
seção 5. `SEED = 42` em tudo.

!!! todo "Correções do EDA"

    Liste aqui o que a correção do EDA apontou e como foi resolvido (ou "nenhuma").

!!! todo "Métrica principal e critério de sucesso (definidos antes de treinar)"

    - **Métrica principal:** TODO (ex.: MAE na escala original do retorno).
    - **Critério de sucesso:** TODO (ex.: "MAE pelo menos X% menor que o do melhor baseline").

## 1. Baselines

Avaliados na **validação**, com o mesmo pipeline.

| Modelo | MAE | RMSE | $R^2$ |
|--------|-----|------|-------|
| `DummyRegressor(strategy="mean")` | | | |
| `Ridge` | | | |

!!! todo "Leitura"

    O que os baselines dizem sobre o sinal disponível nas quatro features.

## 2. A rede

MLP em PyTorch, com código em `code/`. Saída: **1 unidade linear**.

``` mermaid
flowchart LR
    x["Entrada<br/>(4 features)"] --> h1["Oculta 1"]
    h1 --> h2["Oculta 2"]
    h2 --> out["Saída<br/>(1, linear)"]
```

| | Escolha | Por quê |
|---|---------|---------|
| Camadas × unidades | | |
| Ativação | | |
| Inicialização | | |
| Saída | 1 unidade linear | alvo contínuo, sem limite de faixa |
| Perda | | justificar pela cauda pesada do alvo ([EDA §6](../eda/index.md#6-qualidade-dos-dados)) |
| Nº de parâmetros vs. linhas de treino | | |

## 3. Treinamento

| | Escolha | Por quê |
|---|---------|---------|
| Otimizador | | |
| Learning rate | | |
| Batch size | | |
| Épocas (máx.) / paciência | | |

Early stopping na validação, restaurando os melhores pesos.

!!! todo "Figura: curvas de treino e validação por época"

    Ler a curva: converge? Em que época começa o *overfitting*?

## 4. Regularização e hiperparâmetros

### A. Técnicas de regularização

| Configuração | Val. MAE | Val. RMSE | Época de parada |
|--------------|----------|-----------|-----------------|
| Sem regularização | | | |
| Técnica 1 (TODO) | | | |
| Técnica 2 (TODO) | | | |

### B. Busca de hiperparâmetros

Pelo menos 8 configurações, escolhidas pela validação, em ordem crescente da métrica principal.

| # | lr | Camadas × unidades | Regularização | Val. métrica principal |
|---|----|--------------------|---------------|------------------------|
| 1 | | | | |

## 5. Avaliação no teste

Configuração escolhida retreinada e avaliada **uma vez** no teste, com 3 seeds (média ± desvio).

| Modelo | MAE | RMSE | $R^2$ |
|--------|-----|------|-------|
| `DummyRegressor` | | | |
| `Ridge` | | | |
| MLP (3 seeds) | | | |

MAPE não é usado: o alvo fica perto de zero.

!!! todo "Figuras: previsto × real e resíduo × previsto"

!!! todo "Leitura"

    O critério da seção 0 foi atingido? A diferença para o baseline é maior que a variação
    entre seeds?

## 6. Análise de erro

### A. Erro por subgrupo

Sem categóricas no dataset, os subgrupos vêm do tempo e das faixas do alvo (ex.: ano, hora do
dia, minutos com `Volume = 0`, faixas de |alvo|).

| Subgrupo | n | MAE | Hipótese |
|----------|---|-----|----------|
| | | | |

### B. Piores exemplos

### C. Permutation importance

Comparar com o que o EDA sugeriu ([EDA §10B](../eda/index.md#b-reducao-de-dimensionalidade)).

## 7. Conclusão

A rede valeu a pena? Limitações, quem seria afetado por um erro e o que a equipe faria com
mais tempo.

## Results summary

| # | Item | Valor |
|---|------|-------|
| 1 | Dataset, tarefa, alvo e métrica principal | |
| 2 | Linhas de treino · validação · teste | |
| 3 | Melhor baseline e sua métrica no teste | |
| 4 | Rede escolhida: camadas × unidades, nº de parâmetros, regularização | |
| 5 | Otimizador, learning rate, batch e época de parada | |
| 6 | Métrica principal da rede no teste (média ± desvio, 3 seeds) | |
| 7 | Critério de sucesso atingido? | |
| 8 | Subgrupo com maior erro e seu valor | |

## Referências

- Enunciado: [Projects → Classification or Regression](https://insper.github.io/ann-dl/2026.2/projects/supervised/){:target='_blank'}
- Dataset: [Kaggle — mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data){:target='_blank'}
