import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from quicklearning.classification.image import list_files
from quicklearning.classification.image.model import Model
from quicklearning.classification.image import create_folder
from quicklearning.classification.image import remove_folder
from quicklearning.classification.image import create_dataset
from quicklearning.classification.image import remove_bad_prediction_files
from quicklearning.classification.image.callbacks import EarlyStop, LearningRateReducer


def fit(epochs=0, classes=[], data_folder="./images", batch_size=32, validation_split=0.2, predict_retrain_loops=0, verbose=True):
    create_dataset(data_folder=data_folder, classes=classes)
    
    def fit_new_model():
        model = Model(
            _create_model("https://tfhub.dev/google/efficientnet/b0/feature-vector/1", 224, optimizers.RMSprop(), classes),
        )
        model.fit(
            epochs,
            data_folder=data_folder,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                EarlyStop(),
                LearningRateReducer()
            ],
            verbose=verbose
        )
        return model
    model = fit_new_model()
    for _ in range(predict_retrain_loops):
        bad_predictions = remove_bad_prediction_files(model, data_folder, min_accuracy=0.3)
        if bad_predictions > 0:
            model = fit_new_model()
        else:
            return model
    return model

def _create_model(transfer_learning_layer, size, optimizer, classes):
    model = Sequential([
        hub.KerasLayer(transfer_learning_layer, trainable=False, input_shape=[size, size, 3]),
        layers.Dense(len(classes), activation=tf.nn.softmax)
    ])
    if len(classes) <= 2:
        loss = losses.BinaryCrossentropy(from_logits=True)
    else:
        loss = losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.build([None, size, size, 3])
    return model