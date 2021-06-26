from sklearn.datasets import load_files
import numpy as np
import pandas as pd
import os
from metrics import F1_Score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from data_preparation import *
from time import time




def lab3_run_models(x_train, y_train, labels, x_val, y_val, x_test, y_test):
    metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy'), F1_Score('f1_score')]




    first_CNN = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(100, 100, 3)),
        tf.keras.layers.Conv2D(16, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding='valid'),
        tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax')
    ])
    first_CNN.compile(loss='categorical_crossentropy',
                         optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                         metrics=metrics)

    second_CNN = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(100, 100, 3)),
        tf.keras.layers.Conv2D(12, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((4, 4), strides=(4, 4), padding='valid'),
        tf.keras.layers.Conv2D(24, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding='valid'),
        tf.keras.layers.Conv2D(48, (5, 5), strides=(1, 1), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax')
    ])
    second_CNN.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                      metrics=metrics)

    models = {'first_CNN_model': first_CNN,
              'second_CNN_m': second_CNN}

    result_dic = {'name': [],
                  'time': [],
                  'loss': [],
                  'test_acc': [],
                  'test_f1': []}

    for name in models:
        print(models[name].summary())
        model = models[name]
        logdir = os.path.join("lab3_logs", name)
        modeldir = os.path.join("lab3_models", name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        callback = [tensorboard_callback]
        start = time()
        history = model.fit(x=x_train,
                            y=y_train,
                            epochs=5,
                            validation_data=(x_test, y_test),
                            callbacks=callback)
        finish = time()
        model.save(modeldir)
        print(round(time() - start, 5))
        result_dic['name'].append(name)
        result_dic['time'].append(round(time() - start, 5))
        result_dic['loss'].append(history.history['loss'][-1])
        y_pred = np.round(model.predict(x_val))
        f1 = F1_Score()
        acc = tf.keras.metrics.Accuracy()
        f1.update_state(y_val, y_pred)
        acc.update_state(y_val, y_pred)
        result_dic['test_acc'].append(acc.result().numpy())
        result_dic['test_f1'].append(f1.result().numpy())
        print(history.history['val_accuracy'][-1])

    result = pd.DataFrame(result_dic)
    result.to_csv('lab3_logs/result.csv')
    print(result)
    return result