import torch 
from torch import nn  
import pytorch_lightning as L

import numpy as np



# Convolutional Robust Autoencoder designed to process images

'''
En lugar de hacer una clase RAE que tenga self.encoder y self.decoder, es mejor crear el encoder y decoder en clases separadas. Así, se puede
elegir entre guardar solo el encoder, guardar el decoder o guardar ambos. Para tareas de representation learning nos interesará
sobre todo el encoder, el decoder será importante solo para el entrenamiento
'''

class RAE_encoder(nn.Module):


    '''
    Inputs:
    - input_shape: The size of the images
    - inputChannel: The number of input channels (RGB --> 3, Grayscale --> 1)
    - numberOfFilters: The number of convolutional filters applied in the first layer. Subsequent convolutional layer will apply half of the 
    convolutional filters of the last layer (default --> 192)
    - numberOfConvolutions: 
    '''
    def __init__(self, inputChannel=3, numberOfFilters=192):
        super().__init__()

        self.inputChannel = inputChannel
        self.numberOfFilters=numberOfFilters

        # TODO: Estructura dinámica


        # Definimos el encoder

        self.encoder = nn.Sequential(

            # First convolution
            nn.Conv2d(in_channels=self.inputChannel, out_channels=self.numberOfFilters, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

            # Second convolution
            #nn.Conv2d(self.numberOfFilters, self.numberOfFilters/2, (3,3), padding="same"),
            nn.Conv2d(int(self.numberOfFilters), int(self.numberOfFilters/2), (3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

            # Third convolution
            nn.Conv2d(int(self.numberOfFilters/2), int(self.numberOfFilters/4), (3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

            # Fourth convolution
            nn.Conv2d(int(self.numberOfFilters/4), int(self.numberOfFilters/8), (3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

        )

    # Definimos el método forward, que se ejecuta en cada paso de entrenamiento / test
    def forward(self, input_):
        encoded = self.encoder(input_)
        return encoded
    


class RAE_latent(nn.Module):


    '''
    Inputs:
    - input_shape: The size of the images ( [width, height] )
    '''
    def __init__(self, input_shape, numberOfFilters=192):
        super().__init__()

        self.input_shape = input_shape
        self.numberOfFilters = numberOfFilters

        # TODO: 2**numberOfConvolutions /// 2**(numberOfConvolutions-1)
        self.features = int((input_shape[0] / 2**4) * (self.input_shape[1] / 2**4) * self.numberOfFilters/2**3)

        self.latentSpace = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.features, out_features=self.features),

            # El reshape lo hacemos en forward
            

        )

    # Definimos el método forward, que se ejecuta en cada paso de entrenamiento / test
    def forward(self, input_):
        latent = self.latentSpace(input_)

        return latent


        # El reshaped lo hacemos en el decoder para que podamos coger (si queremos) el espacio latente de una muestra
        # reshaped = torch.reshape(latent, ((self.input_shape[0] / 2**4), (self.input_shape[1] / 2**4), (self.numberOfFilters/2**3)))

        # return reshaped
    

class RAE_decoder(nn.Module):

    def __init__(self, input_shape, inputChannel=3, numberOfFilters=192):

        super().__init__()

        self.input_shape = input_shape
        self.numberOfFilters = numberOfFilters
        self.inputChannel=inputChannel

        self.decoder = nn.Sequential(
            
            # First convolution
            nn.ConvTranspose2d(in_channels=int(self.numberOfFilters/8), out_channels=int(self.numberOfFilters/8), kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # Second convolution
            nn.ConvTranspose2d(int(self.numberOfFilters/8), int(self.numberOfFilters/4), (3,3), padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # Third convolution
            nn.ConvTranspose2d(int(self.numberOfFilters/4), int(self.numberOfFilters/2), (3,3), padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # Fourth convolution
            nn.ConvTranspose2d(int(self.numberOfFilters/2), self.numberOfFilters, (3,3), padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # Last convolution to generate the reconstruction with the correct shape
            nn.ConvTranspose2d(self.numberOfFilters, self.inputChannel, (3,3), padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, input_):

        reshaped = torch.reshape(input_, ((self.input_shape[1] / 2**4), (self.input_shape[1] / 2**4), (self.numberOfFilters/2**3)))

        decoded = self.decoder(reshaped)
        return decoded



# # Correntropy loss function used for Robust Autoencoder
# tf_2pi = tf.constant(tf.sqrt(2*np.pi), dtype=tf.float32)

# def robust_kernel(alpha, sigma = 0.2):
#     return 1 / (tf_2pi * sigma) * K.exp(-1 * K.square(alpha) / (2 * sigma * sigma))

# def correntropy(y_true, y_pred):
#     # return -1 * K.sum(robust_kernel(y_pred - y_true))  ## Clasico
#     return -1 * tf.reduce_mean(robust_kernel(y_pred - y_true))  # tf.reduce_mean



class CorrentropyLoss(nn.Module):
    def __init__(self):
        super(CorrentropyLoss, self).__init__()

    
    def robust_kernel(self, alpha, sigma=0.2):
        return 1 / ( (torch.sqrt(2*np.pi)) * sigma) * torch.exp(-1 * torch.square(alpha) / (2 * sigma * sigma))

    def forward(self, y_true, y_pred):
        return -1 * torch.mean(self.robust_kernel(y_pred - y_true))




class RobustAutoencoder(L.LightningModule):

    def __init__(self, encoder, latent, decoder):
        super().__init__()

        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder

        self.robustLoss = CorrentropyLoss()

        

    # Definimos el paso de entrenamiento. Toma como entrada un batch de datos
    def training_step(self, batch, batch_idx):

        # Si el batch size es 16 (especificado en dataloader), aquí llegará una lista (literalmente list) con 16x16
        # elementos, donde lista[0] son las imágenes y lista[1] son las etiquetas

        # Las imágenes (input) de este dataset concreto vienen dadas en el formato [número de canales (1), anchura (28), altura(28)]
        # Como viene en batch, input será del tipo [batch_size, canales, anchura, altura] = [16/32, 1, 28, 28]
        # El propio module es el que se encarga de procesar cada imagen del batch, yo no tengo que hacer nada

        # Tomamos la entrada
        entrada, etiqueta = batch

        # Aquí habría que ver si hay que hacer preprocesamiento adicional (transformar a vector, ...)

        # Pasamos por el encoder
        encoded = self.encoder(entrada)

        # Pasamos por el módulo latent para obtener el espacio latente
        latent = self.latent(encoded)

        # Pasamos por el decoder
        decoded = self.decoder(latent)

        # Calculamos la pérdida (función correntropy para RAE)
        loss = self.robustLoss(entrada, decoded)

        return loss
    

    def test_step(self, batch, batch_idx):

        entrada, etiqueta = batch
        encoded = self.encoder(entrada)
        latent = self.latent(encoded)
        decoded = self.decoder(latent)
        loss = self.robustLoss(entrada, decoded)
        self.log("Test loss (uwu):  ", loss)

    def validation_step(self, batch, batch_idx): 
        entrada, etiqueta = batch
        encoded = self.encoder(entrada)
        latent = self.latent(encoded)
        decoded = self.decoder(latent)
        loss = self.robustLoss(entrada, decoded)
        self.log("Validation loss (uwu):  ", loss)


    # Definimos el optimizador
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)  # Learning rate
        return optimizer
    


