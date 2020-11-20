import io
import os
import shutil
import logging
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import DuckDuckGoImages as ddg
from joblib import Parallel, delayed

def disable_tensorflow_warnings(func):
    def inner(*args, **kwargs):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        x = func(*args, **kwargs)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        return x
    return inner

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def remove_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)

def list_files(path):
    return [name for name in os.listdir(path) if os.path.isfile("{}/{}".format(path, name))]

def list_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir("{}/{}".format(path, name))]

def plot(title, xlabel, ylabel, series=[], leyend=[]):
    for serie in series:
        plt.plot(serie)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(leyend, loc='upper left')

    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def create_dataset(data_folder, classes=[], remove_folder_before_download=False):
    if remove_folder_before_download:
        remove_folder(data_folder)
    create_folder(data_folder)

    if isinstance(classes, list):
        for item in tqdm(classes, desc="downloading dataset", unit="class"):
            folder = '{}/{}'.format(data_folder, item)
            create_folder(folder)
            urls = ddg.get_image_thumbnails_urls(item)
            Parallel(n_jobs=os.cpu_count())(delayed(ddg._download)(url, folder) for url in tqdm(urls, desc=item, unit="img", leave=False))
    elif isinstance(classes, dict):
        for item, query in tqdm(classes.items(), desc="downloading dataset", unit="class"):
            folder = '{}/{}'.format(data_folder, item)
            create_folder(folder)
            if isinstance(query, list):
                urls = []
                for sub_type in query:
                    urls = urls + ddg.get_image_thumbnails_urls(sub_type)
            elif isinstance(query, str):
                urls = ddg.get_image_thumbnails_urls(query)
            Parallel(n_jobs=os.cpu_count())(delayed(ddg._download)(url, folder) for url in tqdm(urls, desc=item, unit="img", leave=False))

def remove_bad_prediction_files(model, data_folder, min_accuracy=0.6):
    removed_files = 0
    with tqdm(model.classes, desc='refine', unit="class") as bar:
        for c in bar:
            files = list_files('{}/{}'.format(data_folder, c))
            for file in tqdm(files, desc='refining class {}'.format(c), unit="img", leave=False):
                predictions = model.predictions(file='{}/{}/{}'.format(data_folder, c, file))
                if predictions[c] < min_accuracy:
                    os.remove('{}/{}/{}'.format(data_folder, c, file))
                    removed_files += 1
                    bar.set_postfix(removed_files=removed_files)
    return removed_files