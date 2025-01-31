import logging
import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision


def get_resnet_18_model(num_classes: int = 29, pretrained: bool = True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class ImageClassifier:
    def __init__(self):
        self.model_ = None
        model_fname = os.path.join(os.path.dirname(__file__), 'model.pth')
        # Check if the model file exists
        if not os.path.isfile(model_fname):
            raise IOError(f'The file "{model_fname}" does not exist!')

        # Load the model
        checkpoint = torch.load(model_fname)
        self.model_ = get_efficentnet_b3_model()
        self.model_.load_state_dict(checkpoint)

        # Set up device and model
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.eval().to(self.device_)

    def _preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).float().to(self.device_)

    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Predict the class of a single image."""
        image_tensor = self._preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model_(image_tensor)
        return outputs / TEMPERATURE

    def predict_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Predict the class of a batch of images."""
        image_tensors = torch.stack([self._preprocess(image) for image in images])
        with torch.no_grad():
            outputs = self.model_(image_tensors)
        return outputs / TEMPERATURE
