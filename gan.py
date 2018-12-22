# GAN to generate art V2

from __future__ import print_function, division
from tensorflow import keras

import matplotlib.pyplot as plt

import sys

import numpy as np

import cv2


def create_dataset():
    imgs = []
    print('Creating dataset')
    for i in range(11024):
        img = cv2.imread('train_1/img_{}.jpg'.format(i))
        if img is not None:
            imgs.append(cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA))
    
    print('Returning dataset')
    return np.array(imgs)

class GAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 10

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = keras.layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = keras.models.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = keras.models.Sequential()

        model.add(keras.layers.Dense(256, input_dim=self.latent_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(3072))
        model.add(keras.layers.Reshape((32*32*3)))
        model.add(keras.layers.UpSampling2D(2))
        model.add(keras.layers.Conv2D(128, (3, 3), strides=1, padding='valid',
                                 activation='relu'))
        model.add(keras.layers.UpSampling2D(2))
        model.add(keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid',
                                 activation='relu'))
        model.add(keras.layers.UpSampling2D(2))
                  
        model.summary()

        noise = keras.layers.Input(shape=(self.latent_dim,))
        img = model(noise)

        return keras.models.Model(noise, img)

    def build_discriminator(self):

        model = keras.models.Sequential()
    
        model.add(keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid',
                                 activation='relu'))
        model.add(keras.layers.MaxPool2D(2))
        model.add(keras.layers.Conv2D(128, (3, 3), strides=1, padding='valid',
                                 activation='relu'))
        model.add(keras.layers.MaxPool2D(2))
        model.add(keras.layers.Flatten()
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = keras.layers.Input(shape=self.img_shape)
        validity = model(img)

        return keras.models.Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = create_dataset()


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose=1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)


        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan_images/%d.png" % epoch)
        plt.close()
        


gan = GAN()
gan.train(epochs=5001, batch_size=64, sample_interval=200)