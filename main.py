import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataset import getDataset
from model import create_model

directory = "images"
fileLocation = "images/vehicle_images/vehicle_images"

training, test = getDataset(directory)

model=create_model(input_shape=(180,180)+(3,), num_classes=2)
