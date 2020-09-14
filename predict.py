#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image

parser = argparse.ArgumentParser ()
parser.add_argument ('image_dir')
parser.add_argument('checkpoint')
parser.add_argument ('--top_k', default = 5, type = int)
parser.add_argument ('--category_names', default = 'label_map.json')

commands = parser.parse_args()
image_path = commands.image_dir
export_path_keras = commands.checkpoint
classes = commands.category_names
top_k = commands.top_k
reloaded = tf.keras.models.load_model(export_path_keras, custom_objects={'KerasLayer': hub.KerasLayer})

with open(classes, 'r') as f:
    class_names = json.load(f)

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    im = np.asarray(im)
    im = process_image(im)
    expand_im = np.expand_dims(im, axis=0)
    p = model.predict(expand_im)
    top_k_values, top_k_indices = tf.nn.top_k(p, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    flower_classes = []
    for idx in top_k_indices[0]:
        flower_classes.append(class_names[str(idx+1)])
    return top_k_values, top_k_indices, flower_classes

probs, classes, flowers = predict(image_path, reloaded, top_k)
print("Possibility: ",probs)
print("Class: ",classes)
print("Flower: ",flowers)
