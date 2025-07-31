"""
Programa que define el modelo de clasificación basado en ResNet18,
utilizando transferencia de aprendizaje y una capa final personalizada.
"""

# Librerías
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


LEARNING_RATE = 0.001
NUM_CLASS = 2
#MOMENTUM = 0.1 
torch.manual_seed(0)

# Cargar el modelo ResNet18 con sus pesos de imagenet
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Congelar tos parámetros de este modelo preentrenado
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la capa final por una capa lineal con 2 salidas
model.fc = nn.Linear(model.fc.in_features, NUM_CLASS)

# Objetos para función de pérdida y optimizador
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(
    [param for param in model.parameters() if param.requires_grad],
    lr=LEARNING_RATE
)
