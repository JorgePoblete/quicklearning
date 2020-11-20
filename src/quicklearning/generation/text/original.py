import os
import logging
import datetime
import numpy as np
from tqdm.notebook import tqdm
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras import losses

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Text(object):

    def __init__(self, generation_name, models_base_dir, logs_base_dir):
        self.generation_name = generation_name
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tokenizer = None

        self.loss = losses.CategoricalCrossentropy(from_logits=True)
        
        self.disableTensorflowWarnings()
        self.disableGPUMemoryGrowth()
        
        self.model_dir = "{}\\{}\\{}".format(models_base_dir, self.generation_name, self.timestamp)
        self.log_dir = "{}\\{}\\{}".format(logs_base_dir, self.generation_name, self.timestamp)
        self.callbacks = Callbacks(self.model_dir, self.log_dir)
       
    def preprocess(self, dataframe, top=None, max_n_gram=None):
        self.tokenizer = Tokenizer(num_words=top)
        self.tokenizer.fit_on_texts(dataframe.iloc[:, 0])
        self.total_words = len(self.tokenizer.word_index) + 1 if top is None else top + 1
        
        self.max_sequence_len = 0
        for row in tqdm(dataframe.itertuples(), total=dataframe.shape[0]):
            tokens = self.tokenizer.texts_to_sequences([row[1]])[0]
            length = len(tokens)
            if length > self.max_sequence_len:
                if max_n_gram is not None and length >= max_n_gram:
                    self.max_sequence_len = max_n_gram
                else:
                    self.max_sequence_len = len(tokens)
       
    def generate(self, model, seed, how_many=1):
        for _ in range(how_many):
            token_list = self.tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len - 1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed += " " + output_word
        return seed
        
    def train(self, dataframe, model=None, optimizer=None, epochs=10000, batch_size=32, callbacks=True, verbose=False, cpu_only=False):
        # compile the model using the provided optimizer optimizer
        model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])
        generator = self.DataGenerator(
            dataframe,
            self.tokenizer,
            self.total_words,
            self.max_sequence_len,
            batch_size=batch_size,
        )
        if cpu_only:
            with tf.device('/CPU:0'):
                # do the actual fit of the model
                history = model.fit(
                    generator,
                    epochs=epochs,
                    callbacks=[
                        self.callbacks.Tensorboard(),
                        self.callbacks.LearningRateReducer(),
                        self.callbacks.EarlyStop(),
                    #    self.callbacks.CheckPoint(),
                    ] if callbacks else [],
                    verbose=verbose
                )

                # evaluate the final model accuracy and loss against the validation dataset
                self.evaluate(model, generator)
        else:
            # do the actual fit of the model
            history = model.fit(
                generator,
                epochs=epochs,
                callbacks=[
                    self.callbacks.Tensorboard(),
                    self.callbacks.LearningRateReducer(),
                    self.callbacks.EarlyStop(),
                #    self.callbacks.CheckPoint(),
                ] if callbacks else [],
                verbose=verbose
            )

            # evaluate the final model accuracy and loss against the validation dataset
            self.evaluate(model, generator)

    def evaluate(self, model, data, steps = 100, verbose=False):
        loss, accuracy = model.evaluate(data, steps = steps, verbose=verbose)
        print("loss: {} - accuracy: {}".format(loss, accuracy))
        return loss, accuracy
                
    def disableTensorflowWarnings(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

    def disableGPUMemoryGrowth(self):
        physical_devices = tf.config.list_physical_devices('GPU') 
        try: 
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True) 
        except ValueError as ve:
            logging.getLogger("tensorflow").error("Invalid GPU: {}".format(ve))
        except RuntimeError as re:
            logging.getLogger("tensorflow").error("Runtime already initialized: {}".format(re))


    class DataGenerator(data_utils.Sequence):
        
        def __init__(self, data, tokenizer, num_classes, max_length, batch_size=32):
            self.data = data
            self.tokenizer = tokenizer
            self.num_classes = num_classes
            self.max_length = max_length
            self.batch_size = batch_size
        
        def __len__(self):
            return int(np.floor(len(self.data) / self.batch_size))

        def __getitem__(self, idx):
            batch = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
            
            input_sequences = []
            for row in batch.itertuples():
                token_list = self.tokenizer.texts_to_sequences([row[1]])[0]
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i+1]
                    input_sequences.append(n_gram_sequence)

            input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_length, padding='pre'), dtype="float32")

            xs = input_sequences[:,:-1]
            ys = to_categorical(input_sequences[:,-1], num_classes=self.num_classes)
            
            return (xs, ys)
            
            
class Callbacks:
    
    def __init__(self, model_dir, log_dir):
        self.model_dir = model_dir
        self.log_dir = log_dir

    def Tensorboard(self):
        return tf.keras.callbacks.TensorBoard(
            log_dir = self.log_dir,
            profile_batch = '10,20',
            histogram_freq = 1,
            embeddings_freq = 1,
            write_graph = True,
            write_images = True,
            write_grads = True
        )

    def LearningRateReducer(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'loss',
            factor = 0.5,
            min_delta = 1e-5,
            patience = 20,
            verbose = False
        )

    def EarlyStop(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor = 'loss',
            patience = 30,
            restore_best_weights = True,
            verbose = 1
        )
    
    def CheckPoint(self):
        return tf.keras.callbacks.ModelCheckpoint(
            self.model_dir,
            monitor = 'loss',
            save_best_only = True,
            save_weights_only = False,
            verbose = False
        )