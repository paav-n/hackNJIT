""" For using the premade data set to begin teaching the machine.

"""
# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import CandyColorBuild.py
from candycolorbuild import *

# Import the Candy dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

(train_images, train_labels) = fashion_mnist.load_data()

# Define sorting colors (five)
class_names = ['Red', 'Orange', 'Yellow', 'Green', 'Violet']

# Data preprocessing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

train_images = train_images / 255.0

# Make a model and load from file
model = keras.models.load_model('candy.h5')


# Wait for, then parse input
waiting = True
while(waiting):
    image = open('temp.jpg','r+')
    arrayize(image)
    test_images = open('test_images','r+')
    predictions = model.predict(test_images)
    sendInfo(i, np.argmax(predictions[i]))


# Inputs a jpeg and then creates the map of values
def mapMake(image)
    base_img = tf.image.decode.jpeg(image)
    res_img = tf.image.resize_images(image,[960,540])

# Grabs image, makes it into a model categorize
# Image is 1920x1080, split into subimages 64x64 in an array, [1856,1016]
def arrayize(image):
    mapMake(image)
    return;

# Outputs the information
def sendInfo(coord, item):
    # Send the information
    print(coord + item)
    return;
