import io
import os
import shutil
import logging
import requests
import numpy as np
from PIL import Image
import DuckDuckGoImages as ddg

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from quicklearning.classification.image import plot
from quicklearning.classification.image.lite import Lite
from quicklearning.classification.image import create_folder
from quicklearning.classification.image import disable_tensorflow_warnings

def load_h5_model(model_file="model.h5", labels_file='labels.tf'):
    loaded_model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})

    labels_file = open(labels_file, 'r')
    classes = [c.strip() for c in labels_file.readlines()]

    return Model(loaded_model, classes=classes)

def load_tf_model(model_folder="model.tf", labels_file='labels.tf'):
    loaded_model = tf.keras.models.load_model(model_folder)

    labels_file = open(labels_file, 'r')
    classes = [c.strip() for c in labels_file.readlines()]

    return Model(loaded_model, classes=classes)

class Model(object):
    @disable_tensorflow_warnings
    def __init__(self, model, classes=[]):
        self.model = model
        self.classes = classes
    
    @disable_tensorflow_warnings
    def fit(self, epochs, data_folder="images", batch_size=32, validation_split=0.2, callbacks=[], verbose=True):
        _, w, h, _ = self.model.input_shape
        self.train, self.validation = self._image_data_generator(w, h, data_folder=data_folder, batch_size=batch_size, validation_split=validation_split)
        self.classes = [x for x in self.train.class_indices.keys()]
        self.steps_per_epoch = self.train.n//self.train.batch_size
        self.validation_steps = self.validation.n//self.validation.batch_size

        self.history = self.model.fit(
            self.train,
            epochs=epochs,
            validation_data=self.validation,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )
    
    @disable_tensorflow_warnings
    def _image_data_generator(self, width, height, data_folder="images", batch_size=32, validation_split=0.2):
        generator = ImageDataGenerator(
            rescale=1/255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        train_data = generator.flow_from_directory(
            data_folder, target_size=(width, height), subset='training', batch_size=batch_size
        )
        validation_data = generator.flow_from_directory(
            data_folder, target_size=(width, height), subset='validation', batch_size=batch_size
        )
        return train_data, validation_data

    @disable_tensorflow_warnings
    def save(self, model_file="model.h5", labels_file='labels.tf'):
        self.model.save(model_file, save_format='h5', include_optimizer=True, overwrite=True)
        with open(labels_file,'w') as file:
            file.write("\n".join(self.classes))

    @disable_tensorflow_warnings
    def save_tf(self, folder="model.tf", labels_file='labels.tf'):
        self.model.save(folder, save_format='tf', include_optimizer=True, overwrite=True)
        with open(labels_file,'w') as file:
            file.write("\n".join(self.classes))
    
    @disable_tensorflow_warnings
    def evaluate(self, verbose=False):
        loss, accuracy = self.model.evaluate(self.validation, steps = self.validation_steps, verbose=verbose)
        return {'accuracy': accuracy, 'loss': loss}

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
        
        _, w, h, _ = self.model.input_shape

        img = img.convert('RGB')
        img = img.resize((w, h))
        x = np.array(img)
        x = x / 255.0
        x = x.reshape((1, w, h, 3))

        prediction = self.model.predict(x)
        index = np.argmax(prediction, axis=-1)[0]
        return self.classes[index], prediction[0][index]

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

        _, w, h, _ = self.model.input_shape

        img = img.convert('RGB')
        img = img.resize((w, h))
        x = np.array(img)
        x = x / 255.0
        x = x.reshape((1, w, h, 3))

        prediction = self.model.predict(x)
        predictions = list(zip(self.classes, prediction[0]))
        if sort:
            return sorted(predictions, key=lambda tup: tup[1], reverse=True)
        else:
            return {k: v for (k,v) in predictions}

    def plot_accuracy(self):
        return plot(
            'model accuracy',
            'epoch',
            'accuracy',
            series=[
                self.history.history['accuracy'],
                self.history.history['val_accuracy']
            ],
            leyend=['train', 'val']
        )

    def plot_loss(self):
        return plot(
            'model loss',
            'epoch',
            'loss',
            series=[
                self.history.history['loss'],
                self.history.history['val_loss']
            ],
            leyend=['train', 'val']
        )

    def tflite(self):
        return Lite(self.model, self.classes)