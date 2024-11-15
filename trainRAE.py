from rae import RAE_encoder, RAE_decoder, RAE_latent, RobustAutoencoder



autoencoder = RobustAutoencoder(RAE_encoder(), RAE_latent([256,384,3]), RAE_decoder([3,384,256]))
print(autoencoder)