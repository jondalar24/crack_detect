"""
Programa para manejar los datos que están en 
formato tensor: *.tp y descomprimir el zip alojado
en la URL del curso IBM
"""

from torch.utils.data import Dataset, DataLoader
import torch
import os
import io
import urllib.request
import zipfile
from tqdm import tqdm

SPLIT_INDEX=30000
BATCH_SIZE = 100

# Barra de progreso personalizada
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(url, zip_name, extract_to="./"):
    """
    Descarga un archivo .zip desde una URL y lo descomprime
    en extract_to
    """
    # Crear directorio si no existe
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    zip_path = os.path.join(extract_to, zip_name)

    # Descargar el archivo si no está
    if not os.path.exists(zip_path):
        print(f"Descargando {zip_name} ...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=zip_name) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)

        print("Descarga Completada!! ")
    else:
        print(f"{zip_name} ya descargado")
    
    # Descomprimir si no está extraido
    folder = os.path.join(extract_to, zip_name.replace('.zip',''))
    if not os.path.exists(folder):
        print(f"Extrayendo {zip_name}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extracción completada")
    else:
        print("Los datos ya están extraídos")

def get_dataloaders(batch_size=BATCH_SIZE):
    train_dataset = DataCrack(train=True)
    val_dataset = DataCrack(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_dataset, val_dataset

    

class DataCrack(Dataset):
    """
    Dataset personalizado para cargar tensores .pt de imágenes
    etiquetadas como positivas (con grieta) o negativas.
    """
    def __init__(self, transform=None, train=True, split_index=SPLIT_INDEX):
        directory = "./"
        positive = "Positive_tensors"
        negative = "Negative_tensors"

        # Obtener rutas de los tensores
        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)
        # lista con las rutas a cada archivo positivo
        positive_files = []
        for file in os.listdir(positive_file_path):
            if file.endswith(".pt"):
                positive_files.append(os.path.join(positive_file_path,file))
        #lista con las rutas a cada archivo negativo
        negative_files = []
        for file in os.listdir(negative_file_path):
            if file.endswith(".pt"):
                negative_files.append(os.path.join(negative_file_path,file))
        
        # Intercalar archivos positivos y negativos
        n_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * n_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        # Etiquetas 1=positivo (hay brecha); 0=negativo (no hay)
        self.Y = torch.zeros([n_samples], dtype=torch.long)
        self.Y[::2] = 1

        # División train/val
        if train:
            self.all_files = self.all_files[:split_index]
            self.Y = self.Y[:split_index]
        else:
            self.all_files = self.all_files[split_index:]
            self.Y = self.Y[split_index:]
        
        # Recogemos variables internas
        self.len = len(self.all_files)
        self.transform = transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = torch.load(self.all_files[index])
        label = self.Y[index]

        # Aplica transformación si existe
        if self.transform:
            image = self.transform(image)
        
        return image,label
    

