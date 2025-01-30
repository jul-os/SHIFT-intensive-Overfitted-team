import torchvision.models as models
import torch

def get_resnet_18_model(num_classes: int = 29, pretrained: bool = True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
