import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable GPU usage
os.environ["TF_XLA_FLAGS"] = "--xla_gpu=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class VAEModel:
    def __init__(self, input_shape=(8, 8, 1), latent_dim=4):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder, self.decoder = self.build_vae()
    
    def sampling(self, args):
        """Sampling function for reparameterization trick."""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_vae(self):
        """Builds a Variational Autoencoder model with CNN layers."""
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = Conv2D(64, 3, activation="relu", padding="same", strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)

        # Latent space
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # Decoder
        decoder_input = Input(shape=(self.latent_dim,))
        x = Dense(16, activation="relu")(decoder_input)
        x = Dense(4 * 4 * 64, activation="relu")(x)
        x = Reshape((4, 4, 64))(x)
        x = Conv2DTranspose(64, 3, activation="relu", padding="same", strides=(2, 2))(x)
        outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        # Build models
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        decoder = Model(decoder_input, outputs, name="decoder")

        return encoder, decoder
    
    def generate_grid(self, level="easy", shape=(3, 5)):
        # Generate a game level grid based on the VAE model and difficulty level.   
        difficulty_mapping = {"easy": 1, "medium": 5, "hard": 10}
        noise = np.random.normal(size=(1, self.latent_dim)) * difficulty_mapping[level]

        # Generate a grid of the given shape
        generated_grid = self.decoder.predict(noise)
        
        # Resize output to the correct dimensions
        resized_grid = tf.image.resize(generated_grid, size=shape).numpy()
        
        return (resized_grid[0, :, :, 0] * 255).astype(int)  # Convert to pixel values