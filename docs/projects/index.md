# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Theo Ribeiro França | | [TheoRibeiroFranca](https://github.com/TheoRibeiroFranca){:target='_blank'} |
| Lucas Sato Coelho de Oliveira | | [lucassatoco](https://github.com/lucassatoco){:target='_blank'} |
| Rafael Kauffmann | | [RafaelKauffmann](https://github.com/RafaelKauffmann){:target='_blank'} |

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página | Prazo | Peso |
|---|---------|--------|-------|------|
| 1 | EDA | [EDA](eda/index.md) | 08.out.2026 | 20% |
| 2 | Regressão | [Regressão](regression/index.md) | 13.nov.2026 | 60% |
| 3 | Generativo | [Generativo](generative/index.md) | 20.nov.2026 | 20% |

Datas e pesos da edição 2026.2 — confira no
[overview](https://insper.github.io/ann-dl/2026.2/){:target='_blank'}. A **prova de projeto** é em 19.nov.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | Bitcoin Historical Data (`btcusd_1-min_data.csv`), exchange Bitstamp |
| **Fonte (URL)** | [Kaggle — mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data){:target='_blank'} |
| **Licença / termos de uso** | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/){:target='_blank'} |
| **Amostras** | 7 764 703 minutos (2012-01-01 → 2026-10-06) |
| **Features** | Brutas: `Open`, `High`, `Low`, `Volume`. Após o pré-processamento: `ret_1`, `amplitude`, `direcao`, `log1p(Volume)` |
| **Variável alvo** | Retorno logarítmico do minuto seguinte, `log(Close[t+1] / Close[t])` |
| **Tarefa escolhida** | Regressão |

É uma série longa (quase 15 anos), limpa e pública, com milhões de amostras, o suficiente
para treinar uma rede neural. O que a torna difícil em vez de trivial: prever o `Close` em
nível é vazamento (os preços do mesmo minuto têm correlação 1,0 com ele), e o teste está num
regime de preço que o treino nunca viu. Com o alvo em retorno, o problema vira prever a
variação do minuto seguinte, que tem sinal fraco (Spearman de −0,11 com o retorno atual).
Detalhes no [EDA](eda/index.md).

## Status

- [ ] **1. EDA**
- [ ] **2. Regressão**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-10-07 | Dataset: BTC/USD de 1 minuto da Bitstamp; tarefa de **regressão** | Série longa e limpa, alvo contínuo natural ([EDA §1](eda/index.md#1-dataset)) |
| 2026-10-07 | Split **temporal** com corte em 2024-01-01 (81% treino / 19% teste) | Split aleatório colocaria minutos vizinhos, quase idênticos, em treino e teste ([EDA §9](eda/index.md#9-estrategia-de-split)) |
| 2026-10-08 | Nenhum outlier removido; `Volume` entra como `log1p(Volume)` | Os preços "outliers" são o regime recente (teste); o `Volume` tem assimetria de 28,5 e 16,9% de zeros ([EDA §6](eda/index.md#6-qualidade-dos-dados)) |
| 2026-10-12 | `Open`, `High` e `Low` do mesmo minuto não entram em nível | Correlação de 1,0 com o `Close`: vazamento ([EDA §5](eda/index.md#5-analise-bivariada-e-correlacoes)) |
| 2026-10-14 | Alvo muda de `Close` para o retorno do minuto seguinte; entradas viram retornos relativos | 61,4% do teste tem preço acima do máximo do treino; retornos não dependem do patamar de preço ([EDA §10](eda/index.md#10-pre-processamento)) |
| 2026-10-14 | Sem redução de dimensionalidade no pipeline | Só 4 features; PCA, t-SNE e UMAP usados apenas para inspeção ([EDA §10](eda/index.md#10-pre-processamento)) |
| 2026-11-12 | Versão do CSV: arquivo local com fim em 2026-06-26 00:22 UTC (7 617 622 linhas; teste com 1 306 102) | O EDA usou arquivo mais longo (7 764 703 minutos, até 2026-10-06). Mesmo corte em 2024-01-01, mesmas regras de split e `SEED = 42` |
| 2026-11-12 | Métrica principal da regressão: RMSE na escala original do log-retorno, com MAE e R² ao lado. Critério fixado antes do treino: RMSE de teste da rede (média de 3 seeds) pelo menos 0,5% menor que o do melhor baseline da validação, e diferença maior que 2 desvios entre seeds | RMSE é a métrica em que uma média condicional aparece. O MAE favorece prever zero, porque a mediana é 0 e há muita massa em zero (13% na validação, 30% no treino) |
| 2026-11-12 | Perda Huber (`delta = 1,0`) sobre o alvo padronizado `y / std(y_tr)`; métricas voltam à escala original | Curtose em excesso de 5 697 no treino e máximo de 0,617: com MSE, poucos minutos de 2012–2014 dominariam o gradiente. Sem padronizar, a perda fica em ~1e-6 e os gradientes somem |
| 2026-11-12 | Janela de treino (2012–2022 ou 2018–2022) como hiperparâmetro; validação e teste não mudam | Volume zero cai de 21,9% no treino (2012–2022) para 3,95% em 2023, e a correlação ret_1 × alvo (Spearman) vai de −0,12 no treino a −0,003 na validação. A janela é subconjunto do treino, escolhida só pela validação |
| 2026-11-12 | Features F4 (as 4 do EDA) e F65 (as mesmas 4 nos lags 0 a 15, mais `rv_60`, desvio de `ret_1` nos últimos 60 minutos); só lags do passado, calculados na série inteira antes do split | Extensão do pipeline do EDA. A ACF de \|ret\| ainda vale 0,25 no lag 15 e 0,18 no lag 60 (2023), enquanto a de `ret_1` fica perto de zero após o lag 1. Sem lag que cruze o corte |
