# Redes Neurais Artificiais & Deep Learning

???+ info inline end "Edição"

    **2026.2**

    [Enunciados :material-open-in-new:](https://insper.github.io/ann-dl/){:target='_blank'}

Este site é o **portfólio** das entregas da disciplina. Ele cresce ao longo do semestre:
cada exercício e cada projeto vira um item de menu, e o repositório que o gera é parte
da avaliação — o professor lê o site publicado **e** o repositório (Markdown, código e
histórico do Git).

## Identificação

Quem responde por este repositório. Os **exercícios são individuais**; a equipe do projeto
— que pode ser diferente — fica registrada na [página do projeto](projects/index.md).

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Theo Ribeiro França | theorf@al.insper.edu.br | [@TheoRibeiroFranca](https://github.com/TheoRibeiroFranca){:target='_blank'} |

!!! tip "Como usar este template"

    Este é um **bloco de notas versionado**: registre o que foi feito, o que falta e as
    decisões tomadas, commitando a cada avanço. O prazo de uma entrega é o *timestamp do
    último commit que toca a pasta daquela entrega* — não a hora do formulário nem a da
    publicação no Pages.

    Comece por [Como usar este template](template/index.md).

## Status das entregas

Calendário, pesos e regras da edição **2026.2** — a fonte é o
[overview](https://insper.github.io/ann-dl/2026.2/){:target='_blank'}; se algo divergir, vale o overview.

### Exercícios — individuais · 40% da nota individual

Cada exercício vale 25% deste bloco.

- [x] [Data](exercises/data/index.md) — 10.set · [enunciado :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/exercises/data/){:target='_blank'}
- [ ] [Perceptron](exercises/perceptron/index.md) — 22.set · [enunciado :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/exercises/perceptron/){:target='_blank'}
- [ ] [MLP](exercises/mlp/index.md) — 13.out · [enunciado :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/exercises/mlp/){:target='_blank'}
- [ ] [VAE](exercises/vae/index.md) — 22.out · enunciado ainda não publicado

Os outros 60% da nota individual vêm da **prova final**, em 24.nov.

### [Projeto](projects/index.md) — em equipe

Um projeto, um dataset, três entregas:

- [ ] [EDA](projects/eda/index.md) — 08.out · 20% · [enunciado :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/projects/eda/){:target='_blank'}
- [ ] [Classificação](projects/classification/index.md) **ou** [Regressão](projects/regression/index.md) — 05.nov · 60% · enunciados: [classificação :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/projects/classification/){:target='_blank'} · [regressão :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/projects/regression/){:target='_blank'}
- [ ] [Generativo](projects/generative/index.md) — 20.nov · 20% · [enunciado :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/projects/generative/){:target='_blank'}

!!! warning "As duas notas precisam de 5"

    A nota final é a média entre individual e equipe **só se ambas alcançarem 5**; caso
    contrário, é o **mínimo** das duas. E a nota de equipe é o mínimo entre o projeto e a
    **prova de projeto** (19.nov) — a prova não soma, ela limita.

    Toda nota está sujeita a **defesa oral**: resultado negativo na defesa zera a nota
    correspondente.

## Checklist antes de cada entrega

- [ ] Repositório **público** e o GitHub Pages construindo sem erro.
- [ ] Caminho correto: `docs/exercises/<slug>/index.md` (ou `docs/projects/<slug>/index.md`).
- [ ] *Front matter* com `exercise:` (ou `project:`) e `ai_use:` preenchidos.
- [ ] Títulos espelhando a estrutura do enunciado (`## Exercise N`, `### A`, `### B`, ...).
- [ ] Figuras commitadas em `figures/`, numeradas e exibidas no relatório.
- [ ] Scripts como arquivos reais em `code/`, referenciados via `--8<--`.
- [ ] Tabela **Results summary** completa, sem linhas em branco.
- [ ] Último commit anterior ao prazo.

!!! danger "Escreva para defender"

    As notas da disciplina costumam estar sujeitas a defesa oral, e a nota do projeto, a uma
    prova sobre o próprio projeto. Escreva relatórios que você consiga sustentar meses
    depois — o que inclui entender cada linha do código que está no repositório. Confira as
    regras da sua edição no overview.

!!! danger "Uso de IA"

    O campo `ai_use` é **obrigatório** em toda entrega. Colaborar com IA é permitido;
    não declarar o uso, não. Descreva o que foi gerado, revisado ou depurado com apoio de
    IA — ou escreva `"none"`.
