"""
Matias Cinera - U 6931_8506
CIS-6619
Instructor: Yu Sun
Ta:         Sadman Sakib
Assigment:  Project 1 - Part 3
Food State Classification - NN Model & Trainning 
Note: Assuming code is running locally - NOT COLAB
"""
import tensorflow as tf
from tensorflow.keras import layers
from  matplotlib import pyplot as plt

IMG_SIZE=264
BACH_SIZE = 32
train_img_folder = '/home/matias/Documents/nn/train'
validation_img_folder = '/home/matias/Documents/nn/valid'

# created train & valid ds based on:
# https://www.tensorflow.org/tutorials/images/classification
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

# inception & resnet require too much memory too add to the current network
inception = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE,3),
)
inception.trainable = False

resnet = tf.keras.applications.resnet50.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet')
for layer in resnet.layers:
    layer.trainable = False

# Model definition  
# Partially based on Md Sadman Sakib NN
# https://arxiv.org/abs/2103.02305 ^
model=tf.keras.Sequential([        
    # normalizing the input
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),

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


checkpoint_filepath = '/home/matias/Documents/nn/model_weights'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy'],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data = valid_ds,
    epochs=40,
    batch_size = BACH_SIZE,  # 32, 64 128, 256, 512 --> lower if more memory is needed
    callbacks=[model_checkpoint_callback]
)
# Saving model
model.load_weights(checkpoint_filepath)
filename = '/home/matias/Documents/nn/project1_part3.tf'
model.save(filename)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('training_&_validation_graph.png')

model.evaluate(valid_ds, batch_size=64)