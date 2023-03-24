import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class_names = ['creamy_paste', 'diced', 'floured', 'grated', 'juiced', 
'jullienne', 'mixed', 'other', 'peeled', 'sliced', 'whole']
test_img_folder = '/home/matias/Documents/nn/test'
model_path = '/home/matias/Documents/GitHub/food-state-classification/project1_part2_51-31.tf'
IMG_SIZE=264
BACH_SIZE = 32

def load_model(filename):
    return tf.keras.models.load_model(filename)

model = load_model(model_path)

for img in os.listdir(test_img_folder):
    img_path = os.path.join(test_img_folder, img)
    read_img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    # read_img/255.0

    img_array = image.img_to_array(read_img)
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    print(f'file: {img}, prediction: {class_names[np.argmax(prediction[0])]}')
    

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     test_img_folder,
#     validation_split=0,
#     batch_size = BACH_SIZE,
#     image_size=(IMG_SIZE, IMG_SIZE)
# )

# test_ds = tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3))


# prediction = model.prediction(test_ds)