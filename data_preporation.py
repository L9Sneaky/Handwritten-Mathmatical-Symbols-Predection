import numpy as np
import yaml
import os
import cv2
import random
import pickle
from rich.progress import track
#%%


data_dir = "dataset"
catagories = os.listdir(data_dir)


with open('model/categories.yaml', 'w') as outfile:
    yaml.dump(catagories, outfile, default_flow_style=False)

print(catagories)

#%%


def create_training_data(img_size=64):
    training_data = []
    x_img = []
    y_label = []
    for category in catagories:
        path = os.path.join(data_dir, category)
        class_num = catagories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([np.array(new_array), class_num])

    random.shuffle(training_data)
    for features, label in training_data:
        x_img.append(features)
        y_label.append(label)

    x_img = np.array(x_img)
    y_label = np.array(y_label)
    return x_img, y_label


X, y = create_training_data(64)

#%%
pickle_out = open("pkl/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("pkl/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
