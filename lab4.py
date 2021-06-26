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


encode_lev1 = 1200
encode_lev2 = 300
latent_v = 40
decode_lev1 = encode_lev2
decode_lev2 = encode_lev1
n_epochs = 10





def lab_4_part1(x_train, y_train, labels, x_val, y_val, x_test, y_test, first=False):
    metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy'), F1_Score('f1_score')]
    if(first):
        autoencoder_dir = os.path.join("lab4_models", "1Auto")
        encoder_dir = os.path.join("lab4_models", "1Encoder")
        decoder_dir = os.path.join("lab4_models", "1Decoder")
        classifier_dir = os.path.join("lab4_models", "1Classifier")

        noisy_train_data = x_train
        noisy_test_data = x_test
    else:
        autoencoder_dir = os.path.join("lab4_models", "Auto")
        encoder_dir = os.path.join("lab4_models", "Encoder")
        decoder_dir = os.path.join("lab4_models", "Decoder")
        classifier_dir = os.path.join("lab4_models", "Classifier")

        noisy_train_data = noise(x_train)
        noisy_test_data = noise(x_test)

    input_img = tf.keras.Input(shape=(100, 100, 3))
    flatten = tf.keras.layers.Flatten()(input_img)
    encoded = tf.keras.layers.Dense(encode_lev1, activation='sigmoid')(flatten)
    encoded = tf.keras.layers.Dense(encode_lev2, activation='sigmoid')(encoded)
    encoded = tf.keras.layers.Dense(latent_v, activation='sigmoid')(encoded)

    decoded = tf.keras.layers.Dense(decode_lev1, activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Dense(decode_lev2, activation='sigmoid')(decoded)
    decoded = tf.keras.layers.Dense(30000, activation='sigmoid')(decoded)
    decoded = tf.keras.layers.Reshape((100, 100, 3))(decoded)


    autoencoder = tf.keras.Model(input_img, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    history = autoencoder.fit(x=noisy_train_data,
                            y=x_train,
                            epochs=n_epochs,
                            shuffle=True,
                            validation_data=(noisy_test_data, x_test))
    autoencoder.save(autoencoder_dir)
    encoder = tf.keras.Model(input_img, encoded)
    encoder.save(encoder_dir)
    outputs = tf.keras.layers.Dense(labels.shape[0], activation='softmax')(encoder.output)
    classifier = tf.keras.Model(inputs=encoder.input, outputs=outputs)
    classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=metrics)
    classifier.fit(x=x_train[:(x_train.shape[0] // 10)],
                   y=y_train[:(x_train.shape[0] // 10)],
                   epochs=n_epochs,
                   shuffle=True)
    classifier.save(classifier_dir)
    print(encoder.predict(noisy_test_data).shape)



def lab4_run_models(x_train, y_train, labels, x_val, y_val, x_test, y_test):


    autoencoder = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(encode_lev1, activation='relu'),
        tf.keras.layers.Dense(encode_lev2, activation='relu'),
        tf.keras.layers.Dense(latent_v, activation='relu'),
        tf.keras.layers.Dense(decode_lev1, activation='relu'),
        tf.keras.layers.Dense(decode_lev2, activation='relu'),
        tf.keras.layers.Dense(30000, activation='sigmoid'),
        tf.keras.layers.Reshape((100, 100, 3))
    ])
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error)

    noisy_train_data = noise(x_train)
    noisy_test_data = noise(x_test)

    models = {'AutoEncoder': autoencoder}

    result_dic = {'name': [],
                  'time': [],
                  'loss': []}

    for name in models:

        model = models[name]
        logdir = os.path.join("lab3_logs", name)
        modeldir = os.path.join("lab4_models", name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        callback = [tensorboard_callback]
        start = time()
        history = model.fit(x=noisy_train_data,
                            y=x_train,
                            epochs=n_epochs-2,
                            shuffle=True,
                            validation_data=(noisy_test_data, x_test))
        finish = time()
        model.save(modeldir)
        print(round(time() - start, 5))
        encoder = tf.keras.models.Sequential()
        encoder.add(autoencoder.layers[0])
        encoder.add(autoencoder.layers[1])
        encoder.add(autoencoder.layers[2])
        encoder.add(autoencoder.layers[3])


        print(encoder.predict(x_test).shape)
        encoderdir = os.path.join("lab4_models", "encoder")
        encoder.save(encoderdir)
        classifier = tf.keras.Model(inputs=encoder, outputs=tf.keras.layers.Dense(labels.shape[0]))
        classifier.fit(x=x_train[:x_train.shape[0]/10],
                       y=y_train[:x_train.shape[0]/10],
                       epochs=n_epochs,
                       shuffle=True)
        result_dic['name'].append(name)
        result_dic['time'].append(round(time() - start, 5))
        result_dic['loss'].append(history.history['loss'][-1])
        y_pred = np.round(model.predict(x_val))
    result = pd.DataFrame(result_dic)
    #result.to_csv('lab3_logs/result.csv')
    print(result)
    return result