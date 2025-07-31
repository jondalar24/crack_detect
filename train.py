"""
Programa utilizado para entrenar el modelo con el dataset
"""
from dataset import DataCrack
from model import model, CRITERION, OPTIMIZER
import numpy as np
#from utils import SIZE_OF_IMAGE
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Detectar si hay GPU (Metal en Mac M3) y mover el modelo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Entrenando en dispositivo: {device}")
model.to(device)

# Variables globales
start_time = time.time()
LOSS_PATH = "loss_history.npy"

# función de entrenamiento
def train_model(N_EPOCHS,train_loader, val_loader, train_dataset, val_dataset):
    
    LOSS = []    
    N_val = len(val_dataset)

    for epoch in range(N_EPOCHS):
        print(f"\nÉpoca {epoch+1}/{N_EPOCHS}")
        model.train()
        for image, target in tqdm(train_loader,desc="Entrenando... "):
            image, target = image.to(device), target.to(device)

            OPTIMIZER.zero_grad()            
            yhat = model(image)
            loss = CRITERION(yhat, target)                      
            loss.backward()
            OPTIMIZER.step()
            LOSS.append(loss.item()) 

        # Evaluación
        model.eval()
        correct = 0
        
        with torch.no_grad():
            for image_val, target_val in tqdm(val_loader, "Validando..."):
                image_val, target_val = image_val.to(device), target_val.to(device)                              
                yhat_val = model(image_val)
                yhat_labels = torch.max(yhat_val,1)[1]             
                correct += (yhat_labels == target_val).sum().item()
        accuracy = correct / N_val
        print(f"Época {epoch+1}/{N_EPOCHS} - Accuracy en validación: {accuracy:.4f}")
    
    # Guardar historial de pérdida
    np.save(LOSS_PATH, LOSS)

    # Gráfico
    plt.figure(figsize=(8, 4))
    plt.plot(LOSS, label='Loss')
    plt.xlabel("Iteraiones")
    plt.ylabel("Pérdida")
    plt.title("Pérdida durante el entrenamiento sobre iteraciones")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    elapsed_time = time.time() - start_time
    print(f"Entrenamiento completado en {elapsed_time:.2f} segundos.")


    return accuracy

