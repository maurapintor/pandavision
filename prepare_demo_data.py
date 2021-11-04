import os
import urllib
from urllib.request import urlopen

from config import config

def prepare_images():
    """Downloads a subset of the Imagenet Dataset."""
    img_folder = os.path.join(config.DATA_DIR, "imagenet_subset")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    fname = os.path.join(img_folder, "data.h5")
    if not os.path.exists(fname):
        data_url = 'https://github.com/maurapintor/pandavision/' \
                   'releases/download/v0.1/data.h5'
        urllib.request.urlretrieve(data_url, fname)

def prepare_model():
    """Downloads an ONNX pretrained model."""
    model_folder = os.path.join(config.DATA_DIR, "models")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    fname = os.path.join(model_folder, "model.onnx")
    if not os.path.exists(fname):
        data_url = 'https://github.com/maurapintor/pandavision/' \
                   'releases/download/v0.1/model.onnx'
        urllib.request.urlretrieve(data_url, fname)

if __name__ == '__main__':
    prepare_images()
    prepare_model()