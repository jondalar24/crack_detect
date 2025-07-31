"""
Script principal para descargar los datos, entrenar el modelo y mostrar la precisión final.
"""

from dataset import download_and_extract, get_dataloaders
from images_management import show_data, denormalize, evaluate_model, plot_graphic
from utils import TRANSFORM, SIZE_OF_IMAGE
from train import train_model
import os
import torch
import torch.nn as nn
from model import model
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Hiperparámetros
N_EPOCHS = 3
MODEL_PATH = "resnet18_crack_detector.pt"

url_positive = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip"
url_negative = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip"

def main():
    # Detectar dispositivo
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model.to(device)
    
    #1. Descargar el dataset
    print("Descargando el dataset (Positive)...\n")
    download_and_extract(url_positive, "Positive_tensors.zip")
    print("Descargando el dataset (Positive)...\n")
    download_and_extract(url_negative,"Negative_tensors.zip")

    #2.  Obtener dataloaders y datasets
    print("Preparando DataLoaders...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()


    #3. Entrenar o cargar el modelo 
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print("Modelo cargado desde disco.")

        # Intentar cargar el historial de pérdida
        if os.path.exists("loss_history.npy"):
            LOSS = np.load("loss_history.npy")
            plot_graphic(LOSS)
    else:
        print("Iniciando entrenamiento...")
        accuracy = train_model(N_EPOCHS, train_loader, val_loader, train_dataset, val_dataset)
        print(f"Validation Accuracy: {accuracy:.4f}" )
        # Guardar el modelo entrenado
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Modelo guardado como {MODEL_PATH}")
    
    #4. Evaluar el modelo
    evaluate_model(model, val_loader, device, save_report=True)

if __name__=="__main__":
    main()
