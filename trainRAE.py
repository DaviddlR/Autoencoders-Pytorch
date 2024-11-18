from rae import RAE_encoder, RAE_decoder, RAE_latent, RobustAutoencoder
from DataClass import *

import pytorch_lightning as L
from torch.utils.data import DataLoader



autoencoder = RobustAutoencoder(RAE_encoder(), RAE_latent([3, 256,384]), RAE_decoder([3, 256,384]))
print(autoencoder)


# Cargamos datos de entrada
wwfDataset_train = DataClass("./train", "./wwf.csv")

# Creamos el dataloader
wwfDataloader_train = DataLoader(wwfDataset_train, batch_size=16, num_workers=15)

# Creamos el trainer
trainer = L.Trainer(max_epochs=10)

# Entrenamos
trainer.fit(model=autoencoder, train_dataloaders=wwfDataloader_train)