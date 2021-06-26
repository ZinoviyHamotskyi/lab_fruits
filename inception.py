import tensorflow as tf
from tensorflow.keras import backend, models, layers, optimizers
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from IPython.display import display
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from metrics import F1_Score

np.random.seed(42)

train_dir = os.path.join('Fruits_demo_4', 'Training')
test_dir = os.path.join('Fruits_demo_4', 'Validation')

# Normalize the pixels in the train data images, resize and augment the data.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2, # Zoom in on image by 20%
    horizontal_flip=True) # Flip image horizontally

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')

backend.clear_session()

# InceptionV3 model and use the weights from imagenet
conv_base = InceptionV3(weights = 'imagenet', include_top = False)
InceptionV3_model = conv_base.output
pool = GlobalAveragePooling2D()(InceptionV3_model)
dense_1 = layers.Dense(512, activation = 'relu')(pool)
output = layers.Dense(4, activation = 'softmax')(dense_1)

metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy'), F1_Score('f1_score')]


model_InceptionV3 = models.Model(inputs=conv_base.input, outputs=output)
plot_model(model_InceptionV3)

model_InceptionV3.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=metrics)

logdir = os.path.join("lab3_logs", "inception")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
callback = [tensorboard_callback, EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
history = model_InceptionV3.fit_generator(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    verbose=1,
    callbacks=callback)

