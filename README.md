# Crack Detection with Transfer Learning and ResNet18

Este proyecto aplica técnicas de *deep learning* y *transfer learning* para clasificar imágenes de estructuras de hormigón, determinando si contienen grietas (`Crack`) o no (`No Crack`). Se emplea el modelo **ResNet18** preentrenado sobre ImageNet como extractor de características, ajustando su última capa para adaptarse a nuestro caso de clasificación binaria.

## Requisitos e instalación

### 1. Crear un entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate    # En Mac/Linux
venv\Scripts\activate     # En Windows
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el programa principal

```bash
python main.py
```

Esto descargará automáticamente los datasets, entrenará (o cargará) el modelo y mostrará métricas de evaluación y visualizaciones de errores.

---

## Estructura del Proyecto

- `main.py`: punto de entrada. Controla el flujo completo (descarga, entrenamiento, evaluación).
- `dataset.py`: carga de datos, descarga de ZIPs desde IBM Object Storage, creación de `DataLoader`.
- `model.py`: define el modelo `ResNet18`, congelando sus capas y sustituyendo la capa final por una personalizada.
- `train.py`: rutina de entrenamiento y validación.
- `images_management.py`: funciones de visualización, desnormalización y evaluación de resultados.
- `utils.py`: constantes de normalización y transformaciones.
- `requirements.txt`: dependencias necesarias.

---

## Arquitectura del Modelo

Se utiliza **ResNet18** con sus capas congeladas, añadiendo una capa `Linear(512, 2)` al final para la clasificación binaria. Esto permite aprovechar la potencia de las 17 capas convolucionales entrenadas previamente, adaptando solo la última a nuestro nuevo dominio.

```
ResNet18 (
  (conv1) -> ... -> (layer4) -> (avgpool)
  (fc): Linear(in_features=512, out_features=2, bias=True)  <-- añadida
)
```

---

## Optimización y uso de GPU

Este proyecto detecta automáticamente si hay una GPU disponible. En caso de ejecutarse en un **Mac con chip Apple M1/M2/M3**, utilizará el backend **Metal Performance Shaders (MPS)** de PyTorch, que proporciona aceleración hardware impresionante.

En nuestras pruebas, entrenar 3 épocas en CPU llevó casi 4.5 horas, mientras que usando M3 con MPS se completó en **menos de 10 minutos**.

---

## Resultados

- Modelo entrenado con más de 40.000 imágenes.
- Precisión validada con `sklearn.metrics`.
- Visualización de ejemplos correctamente e incorrectamente clasificados.
- Historial de pérdida guardado para análisis posterior.

---

## Contribución

Estás invitado a experimentar con el código, cambiar el modelo base, probar con tus propias imágenes o mejorar las visualizaciones. Si lo haces, ¡comparte tus resultados! Estaré encantado de ver qué mejoras puedes aportar 

---

## Créditos

Dataset proporcionado por IBM en el curso “Deep Learning with TensorFlow & PyTorch” — Cognitive Class (IBM Developer).
