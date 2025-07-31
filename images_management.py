""""
programa que se encarga de mostrar las imágenes del dataset
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import MEAN, STD
from sklearn.metrics import classification_report

# Función para desnormalizar y crear imagen reproducible
def show_data(data_sample):
    image = denormalize(data_sample[0]).permute(1,2,0).numpy()
    plt.imshow(image)
    plt.title('Etiqueta: '+ str(data_sample[1]))

# Está normalizada para optimizar el trabajo, para visualizar se desnormaliza
def denormalize(img_tensor):
    mean = torch.tensor(MEAN)
    std = torch.tensor(STD)
    return img_tensor * std[:, None, None]+ mean[:, None, None]

# Visualización de errores
def evaluate_model(model, dataloader, device, max_images=6, save_report=False):
    """
    Evalúa el modelo en el conjunto de validación.
    Muestra métricas de clasificación y ejemplos de imágenes clasificadas correctamente y erróneamente.

    Args:
        model: modelo entrenado.
        dataloader: dataloader del conjunto de validación.
        device: cpu o mps/cuda.
        max_images: cuántas imágenes mostrar por clase.
        save_report: si True, guarda el classification_report en un .txt.
    """
    model.eval()
    y_true = []
    y_pred = []
    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

            for img, true_lbl, pred_lbl in zip(images, labels, predictions):
                if len(correct_samples) < max_images and pred_lbl == true_lbl:
                    correct_samples.append((img.cpu(), true_lbl.cpu(), pred_lbl.cpu()))
                elif len(incorrect_samples) < max_images and pred_lbl != true_lbl:
                    incorrect_samples.append((img.cpu(), true_lbl.cpu(), pred_lbl.cpu()))
            if len(correct_samples) >= max_images and len(incorrect_samples) >= max_images:
                break

    report = classification_report(y_true, y_pred, target_names=["Negativo", "Positivo"])
    print(report)

    if save_report:
        with open("classification_report.txt", "w") as f:
            f.write(report)
        print("Reporte guardado en classification_report.txt")

    # Mostrar ejemplos
    def show_examples(samples, title):
        plt.figure(figsize=(12, 6))
        for idx, (img, true_lbl, pred_lbl) in enumerate(samples):
            plt.subplot(2, 3, idx+1)
            img_show = denormalize(img).permute(1, 2, 0).numpy()
            plt.imshow(img_show)
            plt.title(f"V:{true_lbl.item()} - P:{pred_lbl.item()}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    show_examples(correct_samples, "Imágenes bien clasificadas")
    show_examples(incorrect_samples, "Imágenes mal clasificadas")

# Plotear una gráfica
def plot_graphic(LOSS):
    plt.figure(figsize=(8, 4))
    plt.plot(LOSS, label="Loss (previa)")
    plt.xlabel("Iteraciones")
    plt.ylabel("Pérdida")
    plt.title("Historial de pérdida (modelo cargado)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()