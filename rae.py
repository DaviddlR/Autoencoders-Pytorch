import torch  # Pytorch
from torch import nn  # Bloques básicos de redes neuronales
import pytorch_lightning as L


# Convolutional Robust Autoencoder designed to process images

# TODO: hacer Encoder y Decoder como clases separadas. Así puedo entrenar ambos pero quedarme solo con el encoder (que me interesará)
# para proyectos futuros





class RAE_img(nn.Module):

    
    '''
    Inputs:
    - input_shape: The size of the images
    - input_channel: The number of input channels (RGB --> 3, B&W --> 1)
    - numberOfFilters: The number of convolutional filters applied in the first layer. Subsequent convolutional layer will apply half of the 
    convolutional filters of the last layer
    - numberOfConvolutions: 
    '''
    def __init__(self, input_shape, input_channel=3, numberOfFilters=192, numberOfConvolutions=4):

        super().__init__()

        # Define encoder
        self.encoder = nn.Sequential(
            

            # Check sequential list
            # for convolution in range(numberOfConvolutions):

            # First convolution
            nn.Conv2d(in_channels=input_channel, out_channels=numberOfFilters, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

            # Second convolution
            nn.Conv2d(numberOfFilters, numberOfFilters/2, (3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

            # Third convolution
            nn.Conv2d(numberOfFilters/2, numberOfFilters/4, (3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),

            # Fourth convolution
            nn.conv2d(numberOfFilters/4, numberOfFilters/8, (3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2,2), padding="same"),
        )

        # Define latent space
        self.latent = nn.Sequential(

            features = (input_shape[0] / 2**4) * (input_shape[1] / 2**4)

            nn.Flatten(),
            nn.Linear(in_features=, out_features=)

        )

        # Define decoder
        self.decoder = nn.Sequential(

        )





