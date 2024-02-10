from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.models.cloning import sequential
from keras.optimizers import Adamax
import matplotlib.pyplot as plt
import numpy as np


img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows,img_cols,channels)


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def build_generator ():
    noise_shape = (100,)

    model = sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    
