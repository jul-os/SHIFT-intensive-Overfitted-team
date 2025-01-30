import torch
import torch.nn.functional as F

def focal_loss(output, target, gamma=2.0, alpha=None, reduction='mean'):
    """ Реализация Focal Loss """
    
    # Вычисляем вероятности через softmax
    probs = F.softmax(output, dim=1)
    
    # Получаем вероятность правильного класса
    target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)  
    
    # Вычисляем Focal Loss: (-alpha * (1 - p)^gamma * log(p))
    focal_weight = (1 - target_probs) ** gamma
    loss = -focal_weight * target_probs.log()
    
    # Если передан `alpha`, учитываем его
    if alpha is not None:
        alpha_t = alpha[target]
        loss *= alpha_t

    # Возвращаем усредненный или суммарный лосс
    return loss.mean() if reduction == 'mean' else loss.sum()
