import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from PIL import Image

IMG_SIZE=300
BACH_SIZE = 32
train_img_folder = '/home/matias/Documents/nn/train'
validation_img_folder = '/home/matias/Documents/nn/valid'

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_img_folder,
    validation_split=0,
    batch_size = BACH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE)
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    validation_img_folder,
    validation_split=0,
    batch_size = BACH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE)
)


# normalization_layer = tf.keras.layers.Rescaling(1./255)
# normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# normalization_layer = tf.keras.layers.Rescaling(1./255)
# normalized_valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

# valid_ds = valid_ds.map(lambda x,y: (x/255, y))

# Model
model=tf.keras.Sequential([        
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=11, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy'],
)


# print("XDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
# train, epochs=20, validation_data=val
history = model.fit(
    train_ds,
    validation_data = valid_ds,
    epochs=80,
    batch_size = BACH_SIZE,  # 32, 64 128, 256, 512 --> lower if more memory is needed
)
# Saving model & data
filename = '/home/matias/Documents/nn/project1_part2.tf'
model.save(filename)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

model.evaluate(valid_ds, batch_size=64)