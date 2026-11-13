# MLP da entrega 2: Linear -> ReLU -> (Dropout) ... -> Linear(1), saída linear para regressão.
# Aula 5 (MLP): camadas densas com ReLU; sem ativação na saída, porque o alvo é um número real.
# Aula 9 (Deep Networks): init He (kaiming_normal_, sigma^2 = 2/n, fan_in) para ReLU e vieses em zero.
#   O padrão do nn.Linear (kaiming_uniform_(a=sqrt(5))) tem variância 1/(3n), um sexto da He.
# Aula 7 (Regularização): dropout depois da ReLU nas camadas ocultas, nunca na saída.
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # saída linear, 1 unidade
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        # Aula 9: He com fan_in em todas as camadas lineares, vieses em zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (n,)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # conferência: 65 -> 128 -> 64 -> 1 = 65*128+128 + 128*64+64 + 64*1+1
    m = MLP(65, [128, 64])
    print(m)
    print("n_params", n_params(m), "esperado", 65 * 128 + 128 + 128 * 64 + 64 + 64 + 1)
