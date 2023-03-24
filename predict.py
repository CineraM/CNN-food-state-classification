import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Change the path of the folders if needed
test_img_folder = '/home/matias/Documents/nn/test'
model_path = '/home/matias/Documents/GitHub/food-state-classification/project1_part2_51-31.tf'
# Ground truth files, and the results of my network. 
# The results json file is created on this file and stored in the same directory as predict.py 
r_path = '/home/matias/Documents/GitHub/food-state-classification/ground_truth.json'
gt_path = '/home/matias/Documents/GitHub/food-state-classification/results.json'

class_names = ['creamy_paste', 'diced', 'floured', 'grated', 'juiced', 
'jullienne', 'mixed', 'other', 'peeled', 'sliced', 'whole']
IMG_SIZE=264

# load the trained model
def load_model(filename):
    return tf.keras.models.load_model(filename)
model = load_model(model_path)

results = []
for img in os.listdir(test_img_folder):
    img_path = os.path.join(test_img_folder, img)
    read_img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(read_img)
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    # print(f'file: {img}, prediction: {class_names[np.argmax(prediction[0])]}') # debug
    results.append([img, class_names[np.argmax(prediction[0])]])

results.sort()
results_dict = dict(results)
with open('results.json', 'w') as fp:
    json.dump(results_dict, fp, indent=4,separators=(',', ':'))

# checking accuracy with the ground truth
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
print(f'Test Accuracy: {count/500}%')
