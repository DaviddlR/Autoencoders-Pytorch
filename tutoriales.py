import torch  # Pytorch
from torch import nn  # Bloques básicos de redes neuronales
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Torchvision para visión por computador
from torchvision import transforms
from torchvision.datasets import MNIST


import pytorch_lightning as L

import os



### Definimos los modelos en formato TORCH

# Queremos crear un AE básico, así que hacemos un encoder / decoder

# Clase encoder
class Encoder(nn.Module):

    # Constructor
    def __init__(self):
        super().__init__()

        # Definimos el modelo dentro de un bloque sequential
        self.bloqueEncoder = nn.Sequential(
            # Capa lineal (Dense). Input features , Output features
            # Recibimos un vector de 28x28 (ancho x alto) y lo transformamos en un tensor de 64 elementos
            nn.Linear(28 * 28, 64), 

            # Función de activación ReLU
            nn.ReLU(),  

            # Tercera capa lineal
            nn.Linear(64, 3)  
        )

    # Definimos el método forward, que se ejecuta en cada paso de entrenamiento / test
    # Tomando como entrada "x", aplicamos la función "bloqueEncoder"
    def forward(self, x):
        return self.bloqueEncoder(x)
    

# Clase decoder. Mismo esquema que Encoder pero a la inversa
class Decoder(nn.Module):

    # Constructor
    def __init__(self):
        super().__init__()

        # Definimos el modelo
        self.bloqueDecoder = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 28 * 28)
        )

    # Forward
    def forward(self, x):
        return self.bloqueDecoder(x)



### Definimos el módulo Lightning, que definirá cómo interactúa el modelo.

# Esto viene a ser definir los steps



# Creamos la clase Autoencoder como un módulo Lightning
from typing import Any


class LitAutoEncoder(L.LightningModule):

    # Constructor que toma como entrada el encoder y decoder creados como las clases anteriores
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder  # Establecemos el encoder
        self.decoder = decoder  # Establecemos el decoder

    # Definimos el paso de entrenamiento. Toma como entrada un batch de datos
    def training_step(self, batch, batch_idx):

        # Batch es un batch de datos (xd). Si el batch size es 16 (especificado en dataloader), aquí llegará una lista (literalmente list) con 16x16
        # elementos, donde lista[0] son las imágenes y lista[1] son las etiquetas

        # Las imágenes (input) de este dataset concreto vienen dadas en el formato [número de canales (1), anchura (28), altura(28)]
        # Como viene en batch, input será del tipo [batch_size, canales, anchura, altura] = [16/32, 1, 28, 28]
        # El propio module es el que se encarga de procesar cada imagen del batch, yo no tengo que hacer nada


        # Tomamos la entrada
        input, target = batch  
        #print(batch)
        # print(type(batch))
        # print(len(batch))
        # print(len(batch[0]), len(batch[1]))
        # print(type(batch[0]), type(batch[1]))
        # print(batch[1])


        # Hacemos un reshape para transformarlo en vector
        input = input.view(input.size(0), -1)  


        # Lo pasamos por el encoder
        z = self.encoder(input)  

        # Pasamos el resultado por el decoder
        x_hat = self.decoder(z)  

        # Calculamos la pérdida entre la predicción (x_hat) y la entrada al modelo (input) 
        # Si no estuvieramos utilizando un AE, la pérdida se calcularía con "target" (pasa lo mismo en test)
        loss = F.mse_loss(x_hat, input)  

        self.log("Training loss: ", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Devolvemos esa pérdida
        return loss  
    
    # Definimos el paso de test
    def test_step(self, batch, batch_idx):

        # Tomamos la entrada
        input, target = batch

        # Hacemos reshape
        input = input.view(input.size(0), -1) 

        # Pasamos la entrada por encoder
        z = self.encoder(input)

        # Pasamos el resultado por el decoder
        x_hat = self.decoder(z)

        # Calculamos pérdida
        loss = F.mse_loss(x_hat, input)

        # Mostramos resultado
        self.log("test_loss saludos: ", loss)

    # Definimos el paso de validación
    def validation_step(self, batch, batch_idx):

        # Tomamos la entrada
        input, target = batch

        # Hacemos reshape
        input = input.view(input.size(0), -1) 

        # Pasamos la entrada por encoder
        z = self.encoder(input)

        # Pasamos el resultado por el decoder
        x_hat = self.decoder(z)

        # Calculamos pérdida
        loss = F.mse_loss(x_hat, input)

        # Mostramos resultado
        self.log("val_loss saludos: ", loss)



    # Definimos el optimizador
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


train_set = MNIST(root="./MNIST", download=True, train=True ,transform=transforms.ToTensor())
test_set = MNIST(root="./MNIST", download=True, train=False ,transform=transforms.ToTensor())
val_set = MNIST(root="./MNIST", download=True, )



# AQUI SE ESPECIFICA BATCH SIZE
train_loader = DataLoader(train_set, batch_size=16, num_workers=15) # Importante tmb ajustar num_workers para aumentar rendimiento
test_loader = DataLoader(test_set, batch_size=1, num_workers=15)



autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer(max_epochs=10)  # AQUI SE ESPECIFICA MAX EPOCHS
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

