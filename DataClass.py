'''
Importante. Estoy creando un dataset sobre las imágenes porque quiero. Se podría hacer con ImageFolder parecido a keras
flow from directory

https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/ 
'''

# https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/


import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


import os
import pandas as pd


'''
Clase que hereda de dataset.
Hay que crear tres métodos
1. __init__ para inicializar
2. __len__ para decir cuántas imágenes hay
3. __getitem__ para obtener un item
'''
class DataClass(Dataset):

    def __init__(self, img_path, label_path, transform=None, label_transform=None):
        self.img_path = img_path
        self.transform = transform
        self.label_transform = label_transform

        self.img_labels = pd.read_csv(label_path)

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        # Localizamos la imagen y la etiqueta
        imageName = self.img_labels.iloc[index, 0]  # Fila "index", columna 0 (1 es la etiqueta)
        imageLabel = self.img_labels.iloc[index, 1]

        # Leemos la imagen
        image = read_image(imageName)

        # Aplicamos transformaciones si hiciera falta
        if self.transform:
            image = self.transform(image)
        
        if self.label_transform:
            imageLabel = self.label_transform(imageLabel)

        return image, imageLabel
            
        


