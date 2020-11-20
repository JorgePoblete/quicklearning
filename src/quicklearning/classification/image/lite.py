import io
import os
import shutil
import logging
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

from quicklearning.classification.image import create_folder
from quicklearning.classification.image import disable_tensorflow_warnings

def load_tf_lite_model(model_file="model.tflite", labels_file="labels.tf"):
    model_file = open(model_file, "rb")
    loaded_model = model_file.read()

    labels_file = open(labels_file, 'r')
    classes = [c.strip() for c in labels_file.readlines()]

    return Lite(loaded_model, classes=classes, is_tf_lite_model=True)

class Lite(object):
    @disable_tensorflow_warnings
    def __init__(self, model, classes=[], is_tf_lite_model=False):
        self.classes = classes
        if is_tf_lite_model:
            self.model = model
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.model = converter.convert()

        self.interpreter = tf.lite.Interpreter(model_content=self.model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @disable_tensorflow_warnings
    def save(self, model_file="model.tflite", labels_file="labels.tf"):
        with open(model_file, "wb") as file:
            file.write(self.model)
        with open(labels_file,'w') as file:
            file.write("\n".join(self.classes))

    @disable_tensorflow_warnings
    def predict(self, file="", url="", image=None):
        if file is not "":
            img = Image.open(file)
        elif url is not "":
            response = requests.get(url, stream=True, timeout=1.0, allow_redirects=True)
            img = Image.open(io.BytesIO(response.content))
        elif image is not None:
            img = image
        else:
            raise Exception('Input not provided')

        _, w, h, _ = self.input_details[0]['shape']
        input_img = img.convert('RGB')
        input_img = input_img.resize((w, h))
        input_img = np.expand_dims(input_img, axis=0)
        input_img = (np.float32(input_img) - 0.) / 255.
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        index = np.argmax(prediction)

        return self.classes[index], prediction[0][index]

    @disable_tensorflow_warnings
    def predictions(self, file="", url="", image=None, sort=False):
        if file is not "":
            img = Image.open(file)
        elif url is not "":
            response = requests.get(url, stream=True, timeout=1.0, allow_redirects=True)
            img = Image.open(io.BytesIO(response.content))
        elif image is not None:
            img = image
        else:
            raise Exception('Input not provided')

        _, w, h, _ = self.input_details[0]['shape']
        input_img = img.convert('RGB')
        input_img = input_img.resize((w, h))
        input_img = np.expand_dims(input_img, axis=0)
        input_img = (np.float32(input_img) - 0.) / 255.
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = list(zip(self.classes, prediction[0]))
        
        if sort:
            return sorted(predictions, key=lambda tup: tup[1], reverse=True)
        else:
            return {k: v for (k,v) in predictions}