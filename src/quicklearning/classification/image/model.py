import io
import os
import shutil
import logging
import requests
import numpy as np
from PIL import Image
import DuckDuckGoImages as ddg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from quicklearning.classification.image import plot
from quicklearning.classification.image.lite import Lite
from quicklearning.classification.image import create_folder
from quicklearning.classification.image import disable_tensorflow_warnings

class Model(object):
    @disable_tensorflow_warnings
    def __init__(self, model, image_size, classes=[], data_folder="./images", batch_size=32, validation_split=0.2, callbacks=[], verbose=False):
        self.model = model
        self.classes = classes
        self.data_folder = data_folder
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.verbose = verbose

        self.train, self.validation = self._create_image_data_generator()
        self.steps_per_epoch = self.train.n//self.train.batch_size
        self.validation_steps = self.validation.n//self.validation.batch_size
    
    @disable_tensorflow_warnings
    def fit(self, epochs):
        self.history = self.model.fit(
            self.train,
            epochs=epochs,
            validation_data=self.validation,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=self.callbacks,
            verbose=self.verbose
        )

    @disable_tensorflow_warnings
    def save(self, file="model.h5"):
        self.model.save(file, save_format='h5', include_optimizer=True, overwrite=True)

    @disable_tensorflow_warnings
    def save_tf(self, folder="tf_model"):
        self.model.save(folder, save_format='tf', include_optimizer=True, overwrite=True)
    
    @disable_tensorflow_warnings
    def evaluate(self, verbose=False):
        loss, accuracy = self.model.evaluate(self.validation, steps = self.validation_steps, verbose=verbose)
        return loss, accuracy

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

        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        x = np.array(img)
        x = x / 255.0
        x = x.reshape((1, self.image_size, self.image_size, 3))

        prediction = self.model.predict(x)
        index = np.argmax(prediction, axis=-1)[0]
        return self.classes[index], prediction[0][index]

    def accuracy_plot(self):
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

    def loss_plot(self):
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
        return Lite(self.model, self.image_size, self.classes)

    @disable_tensorflow_warnings
    def _create_image_data_generator(self):
        generator = ImageDataGenerator(
            rescale=1/255,
            validation_split=self.validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        train_data = generator.flow_from_directory(
            self.data_folder, target_size=(self.image_size, self.image_size), subset='training', batch_size=self.batch_size
        )
        validation_data = generator.flow_from_directory(
            self.data_folder, target_size=(self.image_size, self.image_size), subset='validation', batch_size=self.batch_size
        )
        return train_data, validation_data