import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
image_directory = 'datasets/'
dataset = []
label = []

no_tumar_images = os.listdir(image_directory + 'no/')
yes_tumar_images = os.listdir(image_directory + 'yes/')

for i, images_name in enumerate(no_tumar_images):
    if images_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + images_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(0)

for i, images_name in enumerate(yes_tumar_images):
    if images_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + images_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, train_size=0.2, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
