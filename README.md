
 ## quicklearning

## Instalation

To install run the following:

```python
pip install quicklearning
```

## automodel usage

First import
```python
from quicklearning.classification.image import quick
```

then run something like
```python
model = quick.fit(10, ['cat', 'dog'], verbose=True)
```

This will create a dataset of images from the image search results from [DuckDuckGo](https://duckduckgo.com), then create a new model based on a pretained model and lastly it will do the actual training of the model.