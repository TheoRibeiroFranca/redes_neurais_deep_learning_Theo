# Tutorial — workflow do GitHub Pages para o exercício Data

Guia de trabalho deste repositório: como escrever, visualizar, validar e publicar a entrega
**Data** do site MkDocs. Não é material de entrega — é a sua referência de processo.

Estado do repositório na última revisão deste tutorial (21.set.2026):

| O quê | Situação |
|---|---|
| Remoto `origin` | `main` em `35f1bde` e `gh-pages` **existe** |
| `main` local | em dia com o remoto, com as alterações da entrega **ainda não commitadas** |
| Site publicado | **no ar** — `/` e `/exercises/data/` respondem 200 |
| `docs/exercises/data/` | **entrega completa**: 3 scripts, 7 figuras, relatório com os 3 exercícios |
| Dataset do Exercício 3 | `data/spaceship-titanic/train.csv` (8 693 × 14), fora de `docs/` |
| `mkdocs build --strict` | passa sem avisos |
| Ambiente `env/` | numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1, mkdocs 1.6.1 |
| Prazo do enunciado | 10.set.2026 — já passou. Confirme com o professor antes de investir tempo. |

A **§3** registra passo a passo como essa entrega foi feita, e a **§4** é a parte reaproveitável:
como converter código solto de notebook para o formato de script que este repositório espera.

---

## 0. O modelo mental: como este site vira uma página web

Você **nunca** edita o HTML nem a branch que o GitHub Pages serve. O fluxo é:

```
você edita docs/*.md  →  git push na main  →  GitHub Actions roda
                                                  mkdocs gh-deploy --force
                                                       ↓ escreve
                                              branch gh-pages (HTML gerado)
                                                       ↓ serve
                                    https://theoribeirofranca.github.io/redes_neurais_deep_learning_Theo/
```

Três consequências práticas:

1. A branch `gh-pages` é **descartável** — reescrita a cada push. Não edite nada lá.
2. Se o site não atualiza, o problema está no run do Actions, não no seu Markdown.
3. O CI instala só o `requirements.txt` (MkDocs). Ele **não** tem numpy/torch, e roda com
   `execute: false` — nada do seu código é executado na publicação. Por isso as figuras
   precisam estar **commitadas** como arquivos (`.png`/`.svg`) e os números **escritos** no
   Markdown.

---

## 1. Setup único (só na primeira vez)

### 1.1 Ativar o ambiente

```shell
cd ~/redes_neurais/redes_neurais_deep_learning_Theo
source ./env/bin/activate
```

O `env/` já existe e já tem tudo instalado. Se algum dia precisar recriar:

```shell
python3 -m venv env && source ./env/bin/activate
python3 -m pip install -r requirements-ml.txt --upgrade
```

Existem dois arquivos de dependências por um motivo deliberado:
[requirements.txt](requirements.txt) (MkDocs, usado pelo CI) e
[requirements-ml.txt](requirements-ml.txt) (inclui o primeiro + numpy/pandas/sklearn/torch,
usado só por você). Manter o torch fora do CI é o que faz cada build do site levar segundos
em vez de minutos.

### 1.2 Publicar o site pela primeira vez

**Já feito** — a branch `gh-pages` existe e o site responde. A sequência abaixo fica como
referência para quem clonar o repositório do zero, ou caso a configuração do Pages se perca.

**a) Enviar o que já existe.**

```shell
git status                 # confirme o que está pendente
git add docs/
git commit -m "docs: atualiza a capa"
git push
```

**b) Verificar se o Actions está ligado.** Abra a aba **Actions** do repositório no GitHub.
Se aparecer o botão *"I understand my workflows, go ahead and enable them"*, clique — forks
vêm com o Actions desligado, e sem isso nenhum push dispara build.

**c) Acompanhar o run.** O push dispara o workflow `ci`
([.github/workflows/main.yaml](.github/workflows/main.yaml)). Leva 1–2 minutos. É ele que
**cria** a branch `gh-pages`.

**d) Apontar o Pages para a `gh-pages`.** Só depois que o run terminar em verde:
**Settings → Pages → Build and deployment → Deploy from a branch** → branch **`gh-pages`**,
pasta **`/ (root)`** → *Save*.

> Apontar o Pages para a `main` publica o Markdown cru, não o site. É o erro mais comum.

**e) Conferir a URL.** O endereço aparece no topo de **Settings → Pages** e precisa ser
idêntico ao `site_url` do [mkdocs.yml](mkdocs.yml) — já está correto:
`https://theoribeirofranca.github.io/redes_neurais_deep_learning_Theo`.

**Se o build falhar com `Permission to ... denied to github-actions[bot]`:**
**Settings → Actions → General → Workflow permissions** → marque **Read and write
permissions** → salve → **Actions → (run que falhou) → Re-run all jobs**.

### 1.3 Repositório público

O Pages em conta gratuita exige repositório público, e a correção lê o site **e** o
repositório. Confirme em **Settings → General → Danger Zone → Change visibility**.

---

## 2. O ciclo de trabalho de cada exercício

Este é o loop que você repete para os Exercícios 1, 2 e 3. Cada volta termina em um commit —
o prazo da entrega é o *timestamp do último commit que toca `docs/exercises/data/`*, então
commite progressivamente, não tudo no fim.

### 2.1 Abra o preview ao vivo (em um terminal separado)

```shell
source ./env/bin/activate
mkdocs serve -o
```

Abre `http://127.0.0.1:8000` e **recarrega sozinho** a cada arquivo salvo. Deixe rodando o dia
inteiro: você escreve o Markdown de um lado e vê o resultado do outro. `Ctrl+C` encerra.

### 2.2 Escreva o script em `code/`

Um arquivo por exercício, em `docs/exercises/data/code/`:

```
docs/exercises/data/code/
  exercise1_point_clouds.py     # Figuras 1, 1b, 2, 3
  exercise2_high_dim.py         # Figuras 4, 5
  exercise3_spaceship.py        # Figura 6
```

O script deve fazer **duas** coisas (a §4 detalha como chegar nesse formato a partir de código
de notebook):

1. **Salvar as figuras** em `docs/exercises/data/figures/` — use o padrão do script-modelo,
   que resolve o caminho a partir do próprio arquivo:

   ```python
   FIGURES = Path(__file__).resolve().parents[1] / "figures"
   FIGURES.mkdir(parents=True, exist_ok=True)
   fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
   plt.close(fig)   # evita vazamento de figuras quando o script gera várias
   ```

2. **Imprimir os números** que vão para o texto e para a tabela *Results summary*. O enunciado
   é explícito: os números solicitados devem aparecer **no texto**, não apenas na saída de
   código. Imprimir no terminal e copiar para o Markdown é o caminho.

Fixe a semente no topo, uma só para todo o relatório:

```python
RNG = np.random.default_rng(42)
```

### 2.3 Rode o script a partir da raiz do repositório

```shell
python docs/exercises/data/code/exercise1_point_clouds.py
```

Sempre da raiz — os caminhos do `--8<--` e o hábito do `mkdocs build` assumem isso.

### 2.4 Escreva o relatório em `docs/exercises/data/index.md`

O arquivo já vem com o esqueleto certo. Quatro coisas são **contrato com a correção**:

**a) Front matter** — obrigatório, no topo, e o `ai_use` preenchido de verdade:

```yaml
---
exercise: data
ai_use: "Claude Code para revisar o código do Exercício 2 e depurar a projeção PCA"
---
```

**b) Títulos espelhando o enunciado** — `## Exercise 1`, `### A`, `### B`, `### C`, na mesma
ordem do enunciado, e a última seção `## Results summary`. A correção percorre o relatório
procurando essa estrutura.

**c) Código incluído, nunca colado.** Sintaxe de snippet do `pymdownx.snippets`, com caminho
relativo à **raiz do repositório**:

````markdown
``` { .python .copy .select linenums='1' title="docs/exercises/data/code/exercise1_point_clouds.py" }
--8<-- "docs/exercises/data/code/exercise1_point_clouds.py"
```
````

Assim relatório e repositório nunca ficam fora de sincronia. Se o arquivo não existir, o
`mkdocs build --strict` falha — o que é bom: você descobre antes do professor.

**d) Figuras com legenda** — caminho relativo à página (`figures/...`, **sem** `docs/`):

```markdown
![Nuvens de pontos das quatro classes gaussianas](figures/fig01-point-clouds.png)
/// caption
**Figura 1** — Dispersão das quatro classes no plano $(x_1, x_2)$ com `scale = 1.0`.
///
```

O enunciado exige título, rótulos de eixo e legenda de classe **dentro** da figura — a
`caption` do MkDocs é adicional, não substitui.

**Matemática:** `$...$` inline e `$$...$$` em bloco (MathJax já configurado):

```markdown
$$
r_{ij} = \frac{\lVert \mu_i - \mu_j \rVert}{\bar{\sigma}_i + \bar{\sigma}_j}
$$
```

### 2.5 Confira no preview

Volte ao navegador em `http://127.0.0.1:8000/exercises/data/`. Verifique: a figura aparece (e
amplia no clique — plugin `glightbox`), o código apareceu inteiro, as fórmulas renderizaram,
a tabela está formatada.

### 2.6 Valide em modo estrito

```shell
mkdocs build --strict
```

**O CI publica mesmo com avisos; o modo estrito não.** Este comando falha em link quebrado,
âncora inexistente e snippet `--8<--` apontando para arquivo ausente. Rode sempre antes do
push. Ele gera a pasta `site/`, que está no `.gitignore` — não commite.

### 2.7 Commit e push

```shell
git add docs/exercises/data/
git commit -m "exercises/data: exercício 1 — nuvens de pontos e taxa de mistura"
git push
```

### 2.8 Confira a publicação

Aba **Actions** → o run precisa terminar com check verde → abra
`https://theoribeirofranca.github.io/redes_neurais_deep_learning_Theo/exercises/data/`.

> **Republicar sem commit novo:** **Actions → ci → Run workflow**. Útil depois de mexer numa
> configuração do GitHub.
>
> **Publicar sem esperar o CI:** `mkdocs gh-deploy` (usa as suas credenciais do Git).
---

## 3. Passo a passo: como a entrega Data foi feita

Esta seção é o registro do que foi efetivamente executado, em 21.set.2026, na ordem em que
aconteceu. Serve de roteiro para as próximas entregas (Perceptron, MLP, VAE), que usam o mesmo
esqueleto.

### 3.1 Ler o enunciado da edição certa — antes de escrever qualquer linha

Este foi o passo que quase deu errado. O repositório traz um notebook de exemplo em
[docs/examples/notebook/data-exercise-1.ipynb](docs/examples/notebook/data-exercise-1.ipynb),
e é natural tratá-lo como o enunciado. **Não é.** Ele é a resolução de uma edição anterior da
disciplina, e o enunciado de 2026.2 mudou em pontos que valem nota:

| | Notebook de exemplo (edição anterior) | Enunciado 2026.2 |
|---|---|---|
| **Ex. 1** | gerar e plotar as 4 nuvens | + 4 fatores de escala $s$, tabela de $r_{ij}$, taxa de mistura, Figuras 2 e 3 |
| **Ex. 2** | um dataset 5D (gaussianas) | **dois** datasets — gaussianas **e** cascas concêntricas — + histogramas de raio |
| **Ex. 3** | imputar, codificar, padronizar | + split estratificado **antes** de tudo, `TotalSpend`, $\log(1+x)$, verificações finais |
| **Semente** | `np.random.seed(42)` | `np.random.default_rng(42)` — exigido explicitamente |

O enunciado real está em
<https://insper.github.io/ann-dl/2026.2/exercises/data/>. Vale baixar e ler inteiro, incluindo
a tabela **Results summary** e os **Grading Criteria** no fim — é lá que está a distribuição de
pontos por item, e ela diz onde vale a pena gastar esforço (o Exercício 3 sozinho vale 4 dos
10 pontos).

> Regra geral: o notebook em `docs/examples/` é exemplo de **formato**, nunca de **conteúdo**.

### 3.2 Trazer o dado que faltava

O Exercício 3 depende do `train.csv` do Spaceship Titanic, que não estava na máquina. O
download oficial é do Kaggle e exige login, então o arquivo veio de um espelho público (o
conteúdo confere: 8 693 linhas × 14 colunas, 805 421 bytes, idêntico em três repositórios
independentes).

**Onde ele foi parar e por quê:**

```
data/spaceship-titanic/train.csv     ← raiz do repositório, FORA de docs/
```

Fora de `docs/` de propósito. Tudo que está sob `docs/` é copiado para dentro do site
publicado; um CSV de 800 KB no site não serve a ninguém. Fora de lá, o arquivo continua
versionado — logo o relatório continua reproduzível por quem clonar — sem inflar o build.

O script resolve o caminho a partir da própria localização, e não do diretório de onde foi
chamado:

```python
ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "data" / "spaceship-titanic" / "train.csv"
```

`parents[4]` porque o script está em `docs/exercises/data/code/` — quatro níveis abaixo da
raiz. E, se o arquivo não existir, ele falha com uma mensagem que diz onde baixar, em vez de
um `FileNotFoundError` cru.

### 3.3 Um script por exercício, três arquivos, nada de notebook

```
docs/exercises/data/code/
  exercise1_point_clouds.py     # Figuras 1, 1b, 2, 3  + r_ij e taxas de mistura
  exercise2_high_dim.py         # Figuras 4, 5         + variâncias, distâncias, acurácia radial
  exercise3_spaceship.py        # Figura 6             + tabelas, shapes, faixas
```

Cada um é autocontido: roda sozinho, salva as suas figuras e imprime os seus números. Não há
módulo compartilhado entre eles de propósito — o relatório inclui cada arquivo inteiro via
`--8<--`, e um `import` de um quarto arquivo deixaria o leitor do relatório sem parte do
código.

### 3.4 Rodar, ler a saída, escrever os números no texto

```shell
source ./env/bin/activate
python docs/exercises/data/code/exercise1_point_clouds.py
python docs/exercises/data/code/exercise2_high_dim.py
python docs/exercises/data/code/exercise3_spaceship.py
```

A saída de cada script é a **fonte** dos números do relatório. Isso é exigência do enunciado
("report the number in the text — not only in the code output") e é o que impede o relatório de
divergir do código.

Um truque que economizou bastante tempo: o `exercise3_spaceship.py` imprime as tabelas grandes
**já em Markdown**, prontas para colar:

```python
print("| Coluna | Ausentes | % do total |")
print("|---|---:|---:|")
for col, row in missing.iterrows():
    print(f"| `{col}` | {int(row['Ausentes'])} | {row['%']:.2f}% |")
```

A tabela de 12 colunas com valores ausentes e a de 5 colunas de gasto foram copiadas
diretamente do terminal para o `index.md`. Zero chance de erro de transcrição.

### 3.5 Validar e publicar

```shell
mkdocs build --strict        # falha em link, âncora e snippet quebrados
git add docs/exercises/data/ docs/index.md data/
git commit -m "exercises/data: entrega completa (exercícios 1, 2 e 3)"
git push
```

E marcar o checkbox da capa em [docs/index.md](docs/index.md): `- [ ]` → `- [x]`.

### 3.6 O que cada exercício produziu

| Exercício | Script | Figuras | Números que foram para o texto |
|---|---|---|---|
| **1** — nuvens 2D | `exercise1_point_clouds.py` | `fig01-point-clouds.png`, `fig01b-decision-boundaries.png`, `fig02-scale-grid.png`, `fig03-mixing-rate.png` | 6 valores de $r_{ij}$, menor = 1.326 no par (0,1); taxas de mistura 0.25 / 7.25 / 19.25 / 48.25 % |
| **2** — 5D | `exercise2_high_dim.py` | `fig04-pca-projections.png`, `fig05-radius-histograms.png` | distâncias 3.228 e 0.266; variâncias 65.97 % e 42.91 %; raios médios; acurácia 100 % do separador radial |
| **3** — Spaceship Titanic | `exercise3_spaceship.py` | `fig06-foodcourt-before-after.png` | balanceamento 50.36 %; tabela de ausentes; `FoodCourt` treino 452.61 / 0.00; shape (6 954, 17); faixas $[-2.00, 3.51]$ e $[-2.00, 3.37]$ |

São **7 figuras**, e não as 6 do enunciado: a Figura 1b é extra. O item C do Exercício 1 pede
para *esboçar* as fronteiras sobre a Figura 1; em vez de desenhar à mão por cima do PNG, o
script calcula a partição de Voronoi das 4 médias e salva um segundo arquivo. A Figura 1
original continua existindo, exatamente como o item A pede.

---

## 4. De função solta de notebook para o modelo deste repositório

Esta é a parte que se repete em toda entrega: pegar código que funciona numa célula e
transformá-lo em um script que o relatório pode incluir. São sete transformações, e elas se
aplicam quase mecanicamente. Os exemplos "antes" são reais — saem do notebook em
`docs/examples/`.

### 4.1 Semente global → gerador explícito

```python
# antes
np.random.seed(42)
np.random.normal(2, 0.8, size=100)

# depois
RNG = np.random.default_rng(42)
RNG.normal(mean, std, size=(N_PER_CLASS, 2))
```

`np.random.seed` mexe em estado global do processo: qualquer biblioteca que sorteie um número
desloca a sua sequência, e o resultado deixa de ser reproduzível sem você perceber. O
`default_rng` devolve um objeto que só você usa. Além disso o enunciado exige essa forma
explicitamente — é um ponto de graça de graça.

### 4.2 Variáveis soltas → constantes no topo

```python
# antes
class0 = {"x": np.random.normal(2, 0.8, size=100), "y": np.random.normal(3, 2.5, size=100)}
class1 = {"x": np.random.normal(5, 1.2, size=100), "y": np.random.normal(6, 1.9, size=100)}
class2 = ...
class3 = ...

# depois
CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}
MEANS = np.array([CLASSES[c]["mean"] for c in sorted(CLASSES)])
STDS  = np.array([CLASSES[c]["std"]  for c in sorted(CLASSES)])
```

Não é questão de estilo. O item B pede as mesmas 4 classes **quatro vezes**, com os desvios
multiplicados por $s$. Com quatro dicionários fixos isso vira quatro blocos copiados; com os
parâmetros separados do sorteio, vira `generate(scale)` e um laço. E o `MEANS` como array
(4, 2) é o que permite calcular a taxa de mistura das 400 distâncias em uma linha de
broadcasting.

> A regra: **separe o que é parâmetro do que é cálculo.** Se o enunciado repete o experimento
> variando alguma coisa, essa coisa tem que ser argumento de função.

### 4.3 `plt.show()` → `savefig` + `close`

```python
# antes
plt.figure(figsize=(8,6))
plt.scatter(class0["x"], class0["y"], label="Class 0", alpha=0.6)
plt.show()

# depois
FIGURES = Path(__file__).resolve().parents[1] / "figures"

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(*X[y == label].T, s=14, alpha=0.7, label=f"Classe {label}")
fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
plt.close(fig)
```

Três mudanças embutidas aí:

1. **`plt.show()` não existe neste fluxo.** O CI roda com `execute: false` e sem interface
   gráfica — a figura precisa ser um arquivo commitado. `savefig` é o passo que faz a imagem
   existir no site.
2. **Caminho derivado de `__file__`, não do `cwd`.** `parents[1]` a partir de
   `docs/exercises/data/code/script.py` é `docs/exercises/data/`, então `figures/` está certo
   independentemente de onde você chamou o script.
3. **API de objeto (`fig, ax`) no lugar da API de estado (`plt.`).** A `plt.` desenha sempre na
   "figura atual", o que funciona numa célula e quebra num script que gera quatro figuras
   seguidas. E `plt.close(fig)` é o que impede o acúmulo de figuras abertas na memória.

### 4.4 Célula que faz tudo → função que devolve valor

```python
# antes — a célula gera, plota e imprime, tudo junto
X = np.vstack((class_A, class_B))
pca = PCA(n_components=2); X_pca = pca.fit_transform(X)
plt.scatter(...); plt.show()
print("Dataset shape:", X.shape)

# depois
def dataset_i() -> tuple[np.ndarray, np.ndarray]: ...
def project(X) -> tuple[np.ndarray, np.ndarray]: ...
def figure4(projections) -> None: ...
def center_distance(X, y) -> float: ...
```

O critério prático: **se o mesmo dado é usado em dois lugares, ele tem que ser retornado por
uma função.** No Exercício 1 a amostra de $s = 1$ aparece na Figura 1, na Figura 1b, na Figura
2 e no cálculo da taxa de mistura. Se cada uma sorteasse a sua, seriam quatro amostras
diferentes — e o relatório estaria descrevendo uma figura com os números de outra. Por isso o
`main()` gera tudo uma vez:

```python
datasets = {scale: generate(scale) for scale in SCALES}
figure1(*datasets[1.0])
figure1b(*datasets[1.0])
figure2(datasets)
```

### 4.5 Número visto na tela → número impresso com formato

```python
# antes
print("Dataset shape:", X.shape)

# depois
print(f"  s = {scale:>4} | taxa de mistura = {rates[scale]:.4f}  ({rates[scale]:.2%})")
```

A saída do script vira texto do relatório. Vale formatar pensando nisso: número de casas
decimais fixo (para o valor no texto bater com o da tabela), e percentual já calculado (para
não errar na hora de multiplicar por 100). Quando a saída é uma tabela, imprima em Markdown —
ver §3.4.

### 4.6 Comentário que diz *o quê* → docstring que diz *por quê*

O código aparece inteiro dentro do relatório, e o leitor é quem corrige. Comentário que repete
o que a linha já diz é ruído:

```python
# ruim
u = v / np.linalg.norm(v, axis=1, keepdims=True)   # normaliza v

# bom
def _unit_directions(n: int) -> np.ndarray:
    """Direções uniformes na esfera unitária de R^5.

    Amostrar v ~ N(0, I_5) e normalizar é o truque padrão: a normal isotrópica
    não privilegia direção alguma, então v/||v|| cai uniformemente na esfera.
    Sortear cada coordenada em [-1, 1] e normalizar concentraria os pontos nas
    diagonais do cubo.
    """
```

Isso não é enfeite: o critério de avaliação separa "implementado" de "implementado **com a
análise**", e metade da análise cabe na docstring da função que a executa. O mesmo vale para as
escolhas do Exercício 3 — cada `SimpleImputer` tem, ao lado, a linha que diz por que aquela
estratégia e não outra.

### 4.7 Script solto → `main()` com guarda

```python
def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ...

if __name__ == "__main__":
    main()
```

O `mkdir(parents=True, exist_ok=True)` dentro do `main()` faz o script funcionar em clone
limpo (a pasta `figures/` pode não existir). E a guarda `__name__` permite importar o arquivo
num REPL para testar uma função isolada sem disparar a geração inteira — útil quando só um
número está estranho.

### 4.8 Resumo da conversão

| No notebook | No modelo | Motivo |
|---|---|---|
| `np.random.seed(42)` | `RNG = np.random.default_rng(42)` | estado global × gerador próprio; exigido pelo enunciado |
| valores dentro da chamada | constantes no topo | o experimento varia parâmetros |
| `plt.show()` | `fig.savefig(...)` + `plt.close(fig)` | o CI não executa nada; a figura é um arquivo |
| `plt.` (API de estado) | `fig, ax = plt.subplots()` | várias figuras por script |
| caminho relativo ao `cwd` | `Path(__file__).resolve().parents[n]` | roda de qualquer diretório |
| célula que faz tudo | funções com retorno | o mesmo dado serve a várias figuras |
| `print(x)` | `print(f"...{x:.4f}...")` | a saída vira o texto do relatório |
| comentário descritivo | docstring com o porquê | o código é lido dentro do relatório |
| código no nível do módulo | `main()` + `if __name__` | importável, e cria `figures/` sozinho |

---

## 5. Armadilhas específicas deste repositório

As três primeiras vinham no template e **já foram corrigidas** na entrega Data — ficam
registradas porque vão reaparecer em Perceptron, MLP e VAE, que partem do mesmo esqueleto.

1. **A figura referenciada não é a que o script gera.** O `index.md` do template aponta para
   `figures/fig01-exemplo.svg` (placeholder), enquanto o script salva outro nome. Ajuste a
   referência ao gerar a figura de verdade.

2. **Não apague o `fig01-exemplo.svg`.** Parece lixo do template, mas
   [docs/examples/index.md](docs/examples/index.md) aponta para
   `../exercises/data/figures/fig01-exemplo.svg`. Apagá-lo derruba o `mkdocs build --strict`
   com `target ... is not found among documentation files`. Isso aconteceu de fato durante
   esta entrega. O arquivo foi restaurado e fica onde está.

3. **A tabela *Results summary* do template não é a do enunciado.** O template lista
   "Separation ratio (scale = 0.5/1.0/2.0)"; o enunciado 2026.2 pede 13 linhas diferentes
   (4 taxas de mistura, menor $r_{ij}$ **e o par**, 2 distâncias entre centros, 2 variâncias
   explicadas, proporção de `Transported`, média/mediana de `FoodCourt` **no treino**, shape
   final, min/max de treino **e** teste). Copie a tabela do enunciado, não a do template.

4. **`\lVert` funciona no Markdown e quebra no matplotlib.** O MathJax da página renderiza
   `$\lVert x \rVert$` sem problema; o *mathtext* do matplotlib não conhece a macro e derruba o
   script com `ParseFatalException: Unknown symbol: \lVert`. Dentro de rótulo de eixo, use
   `r"$\|x\|$"`. Foi exatamente o erro que interrompeu a primeira execução do
   `exercise2_high_dim.py`.

5. **`--8<--` usa caminho da raiz do repositório** (com `docs/` incluído); as figuras usam
   caminho relativo à página (`figures/...`, **sem** `docs/`). É fácil trocar os dois.

6. **O `.gitignore` ignora `site/`, `env/`, `.cache/`, `.vscode/` e `target/`.** Figuras,
   scripts e `data/` **não** são ignorados — mas confira com `git status` que entraram no
   commit. Figura não commitada = imagem quebrada no site publicado.

7. **Arquivo estranho dentro de `docs/` vai para o site.** Há um
   `docs/exercises/data/atividade1.csv` (colunas `Colaborador, Senioridade, Qualidade`) que não
   tem relação com esta entrega. Está fora do controle de versão, então não foi publicado — mas
   se for commitado sem querer, aparece no site. Mova ou apague.

8. **Se um dia optar por notebook:** o build roda com `execute: false`. Rode todas as células,
   **salve com as saídas**, e só então commite — senão o site mostra o notebook sem gráficos.

9. **O notebook de exemplo não é o enunciado.** Ver §3.1 — é a armadilha mais cara das nove.

---

## 6. Referência rápida de comandos

```shell
source ./env/bin/activate                                    # sempre, antes de tudo
mkdocs serve -o                                              # preview ao vivo (terminal à parte)
python docs/exercises/data/code/exercise1_point_clouds.py    # gera figuras + imprime números
python docs/exercises/data/code/exercise2_high_dim.py
python docs/exercises/data/code/exercise3_spaceship.py
mkdocs build --strict                                        # valida — falha em link/snippet quebrado
git add docs/ data/ && git commit -m "..." && git push
```

---

## 7. Checklist antes de considerar a entrega pronta

- [x] Branch `gh-pages` existe e o site abre na URL do **Settings → Pages**
- [x] Repositório **público**
- [x] Front matter com `exercise: data` e `ai_use:` preenchido de verdade
- [x] Títulos `## Exercise 1/2/3` com `### A`, `### B`, `### C` na ordem do enunciado
- [x] Semente `np.random.default_rng(42)` fixa e a mesma em todo o relatório
- [x] Figuras com título, rótulos de eixo e legenda de classe — commitadas em `figures/`
- [x] Scripts como arquivos reais em `code/`, incluídos via `--8<--` (nada de copiar e colar)
- [x] Números citados **no texto**, não só na saída do código
- [x] Tabela `## Results summary` com as 13 linhas **do enunciado**, todas preenchidas
- [x] `mkdocs build --strict` passa sem erro
- [ ] Commit e push feitos
- [ ] Run do Actions verde e a página publicada mostrando o conteúdo novo
- [x] `grep -rn "TROCAR\|Seu Nome" docs/exercises/data/` não retorna nada

---

## 8. Diagnóstico — quando não funcionar

| Sintoma | Causa provável |
|---|---|
| Nenhum run aparece em **Actions** após o push | Workflows desabilitados (fork) — habilite na aba Actions |
| Run falha com `Permission ... denied to github-actions[bot]` | **Settings → Actions → General → Workflow permissions** → *Read and write* |
| **Settings → Pages** não oferece a branch `gh-pages` | O primeiro run ainda não terminou — a branch só existe depois dele |
| Site em 404 | Fonte do Pages não configurada, ou repositório privado |
| Site abre, mas CSS/links quebrados | `site_url` diferente da URL real do Pages |
| Build falha em `Snippet at path ... could not be found` | `--8<--` apontando para arquivo inexistente ou não commitado |
| Build falha em `target ... is not found among documentation files` | Uma imagem ou página referenciada foi apagada — ver §5.2 |
| Figura não aparece no site (mas aparece no `mkdocs serve`) | Arquivo da figura não foi commitado |
| Script morre em `ParseFatalException: Unknown symbol` | Macro LaTeX que o matplotlib não conhece em rótulo de eixo — ver §5.4 |
| `FileNotFoundError` no `exercise3_spaceship.py` | `data/spaceship-titanic/train.csv` ausente — ver §3.2 |
| Notebook sem gráficos | `.ipynb` commitado sem as saídas salvas (`execute: false`) |
