import os
import DuckDuckGoImages as ddg
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
from quicklearning.classification.image.callbacks import EarlyStop, LearningRateReducer


def fit(epochs=0, classes=[], data_folder="./images", batch_size=32, validation_split=0.2, predict_retrain_loops=0, verbose=True):
    _download_data(data_folder, classes)
    
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
        bad_predictions = remove_bad_prediction_files(model, data_folder)
        if bad_predictions > 0:
            model = fit_new_model()
        else:
            return model
    return model

def remove_bad_prediction_files(model, data_folder):
    removed_files = 0
    with tqdm(model.classes, desc='refine', unit="class") as bar:
        for c in bar:
            files = list_files('{}/{}'.format(data_folder, c))
            for file in tqdm(files, desc='refining class {}'.format(c), unit="img", leave=False):
                prediction, _ = model.predict(file='{}/{}/{}'.format(data_folder, c, file))
                if prediction is not c:
                    os.remove('{}/{}/{}'.format(data_folder, c, file))
                    removed_files += 1
                    bar.set_postfix(removed_files=removed_files)
    return removed_files

def _download_data(data_folder, classes):
    remove_folder(data_folder)
    create_folder(data_folder)
    for item in tqdm(classes, desc="downloading dataset", unit="class"):
        folder = '{}/{}'.format(data_folder, item)
        create_folder(folder)
        urls = ddg.get_image_thumbnails_urls(item)
        
        Parallel(n_jobs=os.cpu_count())(delayed(ddg._download)(url, folder) for url in tqdm(urls, desc=item, unit="img", leave=False))

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