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
from keras.applications.inception_resnet_v2 import InceptionResNetV2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')
train_dir = os.path.join(image_dir, 'train')
validation_dir = os.path.join(image_dir, 'test')


#Load the VGG model
resnetV2_conv = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224,224, 3))

# Freeze all the layers except for the last layer: 
for layer in resnetV2_conv.layers[:-4]:
    layer.trainable = False
 

# Create the model
model_resnetV2 = models.Sequential()
 
# Add the vgg convolutional base model
model_resnetV2.add(resnetV2_conv)
 
# Add new layers
model_resnetV2.add(layers.Flatten())
model_resnetV2.add(layers.Dense(1024, activation='relu'))
model_resnetV2.add(layers.Dropout(0.5))
model_resnetV2.add(layers.Dense(6, activation='softmax'))
model_resnetV2.summary()


model_resnetV2.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4), # learning rate should be small so previously learned weights don't vanish
              metrics=['acc'])

# image augmentation for train set and image resizing for validation
from keras.preprocessing.image import ImageDataGenerator
train_datagen_resnetV2 = ImageDataGenerator( # this function will generate augmented images in real time
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen_resnetV2 = ImageDataGenerator(rescale=1./255) # for validation we don't need to augment


 
train_generator_resnetV2 = train_datagen_resnetV2.flow_from_directory( # this function takes images from folders and feeds to Imagedatagenerator
        train_dir,
        target_size=(224,224),
        batch_size=50,
        class_mode='categorical')
 
validation_generator_resnetV2 = validation_datagen_resnetV2.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=10,
        class_mode='categorical',
        shuffle=False)


history_resnetV2 = model_resnetV2.fit_generator(
      train_generator_resnetV2,
      steps_per_epoch=3.5 ,
      epochs=5,
      validation_data=validation_generator_resnetV2,
      validation_steps=3.2,
      verbose=2)


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input



img_path = '3.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
class_labels = ['dang', 'emilia-clarke', 'justin','kit-harington','nikolaj-coster-waldau', 'peter-dinklage']
features = model_resnetV2.predict_classes(x)
print(class_labels[features[0]])


from keras.models import model_from_yaml
model_yaml = model_resnetV2.to_yaml()
with open("model_resnetV2.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model_resnetV2.save_weights("model_resnetV2.h5")
print("Saved model to disk")
 
# later...
 
# load YAML and create model
yaml_file = open('model_resnetV2.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model_resnetV2.h5")



