"""
Programa con utilidades necesarias para el funcionamiento del programa principal
"""
# Librerías
from torchvision import transforms

# Valores de normalización estandar para imagenes RGB (imagenet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SIZE_OF_IMAGE = 3 * 227 * 227

# Pipeline de transformación
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])