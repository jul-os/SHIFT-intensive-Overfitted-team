import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        return logits / self.temperature

def compute_loss(temperature_scaling, logits, targets):
    """Вычисляет потери для температурного шкалирования"""
    scaled_logits = temperature_scaling(logits)
    loss = torch.nn.functional.cross_entropy(scaled_logits, targets)
    return loss

def train_temperature_scaling(logits, targets, initial_temperature=1.0, epoch=100):
    """Обучает параметр температуры для калибровки модели"""
    temperature_scaling = TemperatureScaling(initial_temperature)
    optimizer = optim.LBFGS([temperature_scaling.temperature], lr=0.01, max_iter=epoch)

    for _ in range(epoch):
        optimizer.zero_grad()
        loss = compute_loss(temperature_scaling, logits, targets)
        loss.backward()
        optimizer.step(lambda: compute_loss(temperature_scaling, logits, targets))

    return temperature_scaling.temperature.item()

def apply_temperature_scaling(logits, temperature):
    scaled_logits = logits / temperature
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    return probs