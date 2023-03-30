"""
Matias Cinera - U 6931_8506
CIS-6619
Instructor: Yu Sun
Ta:         Sadman Sakib
Assigment:  Project 1 - Part 2
Food State Classification - Testing script
Note: Assuming code is running locally - NOT COLAB
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Change the path of the folders if needed
test_img_folder =       '/home/matias/Documents/nn/test'
base_model_path =       '/home/matias/Documents/cnn_saved_models/project1_part2.tf'
inception_model_path =  '/home/matias/Documents/cnn_saved_models/project1_part2_inception.tf'
resnet_model_path =     '/home/matias/Documents/cnn_saved_models/project1_part2_resnet.tf'

class_names = ['creamy_paste', 'diced', 'floured', 'grated', 'juiced', 
'jullienne', 'mixed', 'other', 'peeled', 'sliced', 'whole']
IMG_SIZE=264

# load the trained model
def loadModel(filename):
    return tf.keras.models.load_model(filename)

def predictModel(model):
    img_batch = []
    for img in os.listdir(test_img_folder):
        img_path = os.path.join(test_img_folder, img)
        read_img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))

        img_array = image.img_to_array(read_img)
        img_array = np.expand_dims(img_array, axis=0)

        img_batch.append(img_array)
    images = np.vstack(img_batch)
    return model.predict(images, verbose=0)


print("Predicting Base Model...")
model = loadModel(base_model_path)
base_results = predictModel(model)


print("Predicting Inception Model...")
model = loadModel(inception_model_path)
inception_results = predictModel(model)

print("Predicting Resnet Model...")
model = loadModel(resnet_model_path)
resnet_results = predictModel(model)

print(len(base_results), len(inception_results), len(resnet_results))
# sum the results and take the average
ensemble_results = np.array(base_results) + np.array(base_results) + np.array(resnet_results)
ensemble_results/=3

# picking the higest prediciton and assigning it a class based on idex
results = []
index = 0
for img in os.listdir(test_img_folder):
    results.append([img, class_names[np.argmax(ensemble_results[index])]])
    index+=1

results.sort()
results_dict = dict(results)
with open('prediction.json', 'w') as fp:
    json.dump(results_dict, fp, indent=4,separators=(',', ':'))

print("Ensemble prediction stored in 'results.json' ")


# ---- This code calculates the accuracy of the prediction based on ground truth ----
# Ground truth files, and the results of my network. 
# The results json file is created on this file and stored in the same directory as predict.py 

r_path = '/home/matias/Documents/GitHub/food-state-classification/ground_truth.json'
gt_path = '/home/matias/Documents/GitHub/food-state-classification/prediction.json'
results = {}
ground_truth = {}

with open(r_path) as json_file:
    results = json.load(json_file)
with open(gt_path) as json_file:
    ground_truth = json.load(json_file)

count = 0
for key in results:
    if results[key] == ground_truth[key]:
        count+=1

print(f'Samples Correctly classified: {count} of 500')
print(f'Test Accuracy: {(count/500)*100}%')


# results = []
# for img in os.listdir(test_img_folder):
#     img_path = os.path.join(test_img_folder, img)
#     read_img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))

#     img_array = image.img_to_array(read_img)
#     img_batch = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_batch, verbose=0)
#     print(f'file: {img}, prediction: {class_names[np.argmax(prediction[0])]}') # debug
#     results.append([img, class_names[np.argmax(prediction[0])]])