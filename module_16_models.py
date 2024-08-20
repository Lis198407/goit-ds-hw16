import tensorflow as tf
import numpy as np
import keras as k
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

def add_padding(images, padding_width=2):
    num_images = images.shape[0]
    height, width, channels = images.shape[1], images.shape[2], images.shape[3]
    padded_height = height + 2 * padding_width
    padded_width = width + 2 * padding_width

    padded_images = np.zeros((num_images, padded_height, padded_width, channels), dtype=images.dtype)
    padded_images[:, padding_width:padding_width + height, padding_width:padding_width + width, :] = images
    return padded_images

def get_data_for_cnn():
    (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((60000,28,28,1))
    test_images = test_images.reshape((10000,28,28,1))

    train_images = train_images.astype('float32') / 255  # Normalize data
    test_images = test_images.astype('float32') / 255  # Normalize data

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    train_images_pad = add_padding(train_images) #, padding_width=2
    test_images_pad = add_padding(test_images) #, padding_width=2
    return (train_images_pad, train_labels), (test_images_pad, test_labels)

def get_data_for_vgg16(batch_size = 64):

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

    # Add padding. from 28*28 -> 32*32
    x_train_padded = add_padding(x_train, padding_width=2)
    x_test_padded = add_padding(x_test, padding_width=2)

    #numpy to tensors
    x_train_tf = tf.convert_to_tensor(x_train_padded, dtype=tf.float32)
    x_test_tf = tf.convert_to_tensor(x_test_padded, dtype=tf.float32)

    # grayscale to RGB
    x_train_rgb_tf = tf.image.grayscale_to_rgb(x_train_tf)
    x_test_rgb_tf = tf.image.grayscale_to_rgb(x_test_tf)

    # labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=10)
    y_test_one_hot = to_categorical(y_test, num_classes=10)

    # Convert one-hot encoded labels to tensors
    y_train_tf = tf.convert_to_tensor(y_train_one_hot, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test_one_hot, dtype=tf.float32)

    # Create TensorFlow datasets with batches
    train_data = tf.data.Dataset.from_tensor_slices((x_train_rgb_tf, y_train_tf)).shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((x_test_rgb_tf, y_test_tf)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return (train_data, test_data), (x_test_padded, y_test)

def CNN_model(activation_function_l1= 'relu', dropout=0.5, n_hidden_1 = 128, activation_output= 'softmax', optimizer= 'adam', loss_func="categorical_crossentropy", metrics = "accuracy"):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation=activation_function_l1, input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32,(3,3),activation=activation_function_l1),
        layers.Conv2D(32,(3,3),activation=activation_function_l1),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3,3),activation=activation_function_l1),
        layers.MaxPooling2D((2,2)),
               
        layers.Flatten(),
        layers.Dense(n_hidden_1,activation=activation_function_l1),
        layers.Dropout(dropout),
        layers.Dense(n_hidden_1,activation=activation_function_l1),
        layers.Dropout(dropout),
        layers.Dense(10,activation=activation_output),
    ])

    model.compile(
        optimizer= optimizer,
        loss=loss_func,
        metrics=[metrics]
    )
    return model

def VGG16_model(activation_function_l1= 'relu', dropout=0.5, n_hidden_1 = 128, activation_output= 'softmax', optimizer= 'adam', loss_func="categorical_crossentropy", metrics = "accuracy"):
    
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    conv_base.trainable = False

    model = models.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(n_hidden_1, activation=activation_function_l1),
        layers.Dropout(dropout),
        layers.Dense(n_hidden_1, activation=activation_function_l1),
        layers.Dropout(dropout),
        layers.Dense(10, activation=activation_output)
    ])

    model.compile(
        loss=loss_func,
        optimizer=optimizer,
        metrics=[metrics]
    )

    return model