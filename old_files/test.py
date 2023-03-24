import json

results = {}
ground_truth = {}

r_path = '/home/matias/Documents/GitHub/food-state-classification/ground_truth.json'
gt_path = '/home/matias/Documents/GitHub/food-state-classification/result.json'

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
