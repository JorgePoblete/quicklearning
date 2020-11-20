from quicklearning.classification.image.model import Model


def fit(model, epochs=0, data_folder="./images", batch_size=32, validation_split=0.2, callbacks=[], verbose=True):    
        model = Model(model)
        model.fit(
            epochs,
            data_folder=data_folder,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        return model