import io
import os
import shutil
import logging
from PIL import Image
import matplotlib.pyplot as plt

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