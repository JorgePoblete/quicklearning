import tensorflow as tf

from quicklearning.classification.image import create_folder

def LearningRateReducer():
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,
        min_delta = 1e-5,
        patience = 5,
        verbose = False
    )

def EarlyStop():
    return tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        restore_best_weights = True,
        verbose = 1
    )

def CheckPoint(folder):
    create_folder(folder)
    return tf.keras.callbacks.ModelCheckpoint(
        folder,
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = False,
        verbose = False
    )