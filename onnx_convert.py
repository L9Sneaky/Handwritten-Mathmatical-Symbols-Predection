import tensorflow as tf
import onnx
import tf2onnx
import cv2
import numpy as np


# %%


model = tf.keras.models.load_model("model/model.h5")
model.load_weights('model/modelW.h5')

spec = (tf.TensorSpec((None, 64, 64, 3), tf.double, name="input"),)
model = tf2onnx.convert.from_keras(model, input_signature=spec, opset=17, output_path='model/model.onnx')

# %%

import onnxruntime as rt
import yaml
session = rt.InferenceSession('model/model.onnx')

inputDetails = session.get_inputs()

def prep_img(img_path):
    IMG_SIZE = 64
    x = cv2.imread(img_path)
    x = x / 255.0
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(x, axis=0)
    return x

with open("model/categories.yaml", 'r') as stream:
    categories = yaml.safe_load(stream)

im = prep_img('Untitled.png')
y = session.run(None, {inputDetails[0].name: im})
result = categories[int(np.argmax(y))]
result
