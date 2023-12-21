import numpy as np

import random
import os
import tensorflow as tf
from tqdm import tqdm
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from time import time
from sklearn.model_selection import train_test_split

def create_dnn():
    # Network Parameters for runs
    final_start_time = time()

    start_time = time()
    # Network Parameters
    input_shape = (64, 64, 3)
    batch_size = 64
    epochs = 100

    print(f'Input shape is: {input_shape}\nBatch size is: {batch_size}\nNumber of epochs is: {epochs}')
    # Parameters that stay constant
    kernel_size = 3
    latent_dim = 256

    # PATH = '/app/input/blur-dataset/'
    PATH = '../input/blur-dataset/'
    good_frames = PATH + 'sharp'
    bad_frames = PATH + 'defocused_blurred'

    clean_frames = []
    for file in tqdm(sorted(os.listdir(good_frames))):
        if any(extension in file for extension in ['.jpg', 'jpeg', '.JPG']):
            image = tf.keras.preprocessing.image.load_img(good_frames + '/' + file, target_size=(input_shape[0],input_shape[1]))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
            clean_frames.append(image)

    clean_frames = np.array(clean_frames)
    blurry_frames = []
    for file in tqdm(sorted(os.listdir(bad_frames))):
        if any(extension in file for extension in ['.jpg', 'jpeg', '.JPG']):
            image = tf.keras.preprocessing.image.load_img(bad_frames + '/' + file, target_size=(input_shape[0],input_shape[1]))
            image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
            blurry_frames.append(image)

    blurry_frames = np.array(blurry_frames)

    print("\n")
    print("\n")
    print("\n")
    print(f'Clean frames shape: {clean_frames.shape}')
    print(f'Blurry frames shape: {blurry_frames.shape}')
    print("\n")

    seed = 21
    random.seed = seed
    np.random.seed = seed

    X = clean_frames
    y = blurry_frames
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print()
    print(y_train[0].shape)

    # Encoder/Decoder number of CNN layers and filters per layer
    layer_filters = [64, 128, 256]

    # Start the experiment
    inputs = Input(shape = input_shape, name = 'encoder_input')
    x = inputs

    for filters in layer_filters:
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=2,
                activation='relu',
                padding='same')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)


    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    outputs = Conv2DTranspose(filters=3,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    autoencoder.compile(loss='mse', optimizer='adam',metrics=["acc"])

    callbacks = []

    history = autoencoder.fit(x_train,
                            y_train,
                            validation_data=(x_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks)

    end_time = time()

    print(f"Loss: {history.history['loss'][-1]}\n")
    print(f"Val Loss: {history.history['val_loss'][-1]}\n")
    print(f"Acc: {history.history['acc'][-1]}\n")
    print(f"Val Acc: {history.history['val_acc'][-1]}\n")
    print(f"Time: {end_time - start_time}\n")

    final_end_time = time()

    print(final_end_time - final_start_time)

    # Sauvegarde du modèle
    model_save_path = "./train_model.h5"
    # model_save_path = "./train_model.h5"
    autoencoder.save(model_save_path)
