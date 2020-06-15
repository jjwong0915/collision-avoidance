from tensorflow import keras
from tensorflow.keras import layers


def DangerIndexModel(input_shape):
    inputs = keras.Input(shape=input_shape)
    #
    conv_1 = layers.Conv2D(16, 3, activation="relu")(inputs)
    pool_1 = layers.MaxPool2D()(conv_1)
    #
    conv_2 = layers.Conv2D(32, 3, activation="relu")(pool_1)
    pool_2 = layers.MaxPool2D()(conv_2)
    #
    conv_3 = layers.Conv2D(64, 3, activation="relu")(pool_2)
    pool_3 = layers.MaxPool2D()(conv_3)
    #
    flat_0 = layers.Flatten()(pool_3)
    dens_0 = layers.Dense(128, activation="relu")(flat_0)
    dens_1 = layers.Dense(1)(dens_0)
    #
    return keras.Model(inputs=inputs, outputs=dens_1)
