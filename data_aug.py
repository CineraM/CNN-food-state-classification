import tensorflow as tf
from  matplotlib import pyplot as plt

IMG_SIZE=264
BACH_SIZE = 32
train_img_folder = '/home/matias/Documents/nn/train'
validation_img_folder = '/home/matias/Documents/nn/valid'


train_ds , dsinfo = tf.keras.utils.image_dataset_from_directory(
    train_img_folder,
    validation_split=0,
    batch_size = BACH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE)
)

# num_classes = train_ds.features['label'].num_classes
print(dsinfo)