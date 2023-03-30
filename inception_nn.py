"""
Matias Cinera - U 6931_8506
CIS-6619
Instructor: Yu Sun
Ta:         Sadman Sakib
Assigment:  Project 1 - Part 2
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

# https://www.kaggle.com/code/kmader/transfer-learning-with-inceptionv3
# based of^
## inception code ##
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE,3),
)

base_model.trainable = False
add_model = tf.keras.Sequential()
add_model.add(tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
add_model.add(base_model)
add_model.add(tf.keras.layers.GlobalAveragePooling2D())
add_model.add(tf.keras.layers.Dropout(0.5))
add_model.add(tf.keras.layers.Dense(11, activation='softmax'))


model = add_model
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

checkpoint_filepath = '/home/matias/GitHub/food-state-classification/model_weights'
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

history = model.fit(
    train_ds,
    validation_data = valid_ds,
    epochs=25,
    callbacks=[model_checkpoint_callback]
)

# Saving model
model.load_weights(checkpoint_filepath)
filename = '/home/matias/Documents/GitHub/food-state-classification/project1_part2_inception.tf'
model.save(filename)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('training_&_validation_graph_inception.png')