import os
import numpy as np
import cv2
from PIL import Image
import pickle
import os
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')
train_dir = os.path.join(image_dir, 'train')
validation_dir = os.path.join(image_dir, 'test')


from keras.applications import VGG16
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))

# Freeze all the layers except for the last layer: 
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 

# Create the model
model_VGG16 = models.Sequential()
 
# Add the vgg convolutional base model
model_VGG16.add(vgg_conv)
 
# Add new layers
model_VGG16.add(layers.Flatten())
model_VGG16.add(layers.Dense(1024, activation='relu'))
model_VGG16.add(layers.Dropout(0.5))
model_VGG16.add(layers.Dense(6, activation='softmax'))
model_VGG16.summary()


model_VGG16.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4), # learning rate should be small so previously learned weights don't vanish
              metrics=['acc'])

# image augmentation for train set and image resizing for validation
from keras.preprocessing.image import ImageDataGenerator
train_datagen_VGG16 = ImageDataGenerator( # this function will generate augmented images in real time
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen_VGG16 = ImageDataGenerator(rescale=1./255) # for validation we don't need to augment


 
train_generator_VGG16 = train_datagen_VGG16.flow_from_directory( # this function takes images from folders and feeds to Imagedatagenerator
        train_dir,
        target_size=(224,224),
        batch_size=10,
        class_mode='categorical')
 
validation_generator_VGG16 = validation_datagen_VGG16.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=10,
        class_mode='categorical',
        shuffle=False)


history_VGG16 = model_VGG16.fit_generator(
      train_generator_VGG16,
      steps_per_epoch=3.5 ,
      epochs=50,
      validation_data=validation_generator_VGG16,
      validation_steps=3.2,
      verbose=2)


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input



img_path = '3.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
class_labels = ['dang', 'emilia-clarke', 'justin','kit-harington','nikolaj-coster-waldau', 'peter-dinklage']
features = model_VGG16.predict_classes(x)
print(class_labels[features[0]])


from keras.models import model_from_yaml
model_yaml = model_VGG16.to_yaml()
with open("model_VGG16.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model_VGG16.save_weights("model_VGG16.h5")
print("Saved model to disk")
 
# later...
 
# load YAML and create model
yaml_file = open('model_VGG16.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model_VGG16.h5")

