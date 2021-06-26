from sklearn.datasets import load_files
import numpy as np
import pandas as pd
import os
from metrics import F1_Score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from time import time




def lab2_run_models(x_train, y_train, labels, x_test, y_test):
    metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy'), F1_Score('f1_score')]

    # start model with one Softmax layer
    start = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.RandomNormal())
    ])
    start.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  metrics=metrics)

    # NN with 3 tanh activated layers, SGD and Glorot
    only_tanh_MLP = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotNormal()),
        tf.keras.layers.Dense(300, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotNormal()),
        tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotNormal()),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.GlorotNormal())
    ])
    only_tanh_MLP.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                          metrics=metrics)

    # NN with 3 Relu activated layers, SGD and He
    only_ReLU_MLP = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.HeNormal())
    ])
    only_ReLU_MLP.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                          metrics=metrics)

    # NN with 3 Parametrcic Relu activated layers, SGD and He
    parametric_Relu = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(300, kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(100, kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax')
    ])
    parametric_Relu.compile(loss='categorical_crossentropy',
                            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                            metrics=metrics)

    # NN with 3 Relu activated layers, Adagrad and He
    adagrad = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.HeNormal())
    ])
    adagrad.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                    metrics=metrics)

    # NN with 3 Relu activated layers, Adagrad and He with Batch normalization
    batch_normalized = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.BatchNormalization(momentum=0.999),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.BatchNormalization(momentum=0.999),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.BatchNormalization(momentum=0.999),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.HeNormal())
    ])
    batch_normalized.compile(loss='categorical_crossentropy',
                             optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),
                             metrics=metrics)

    # NN with 3 Relu activated layers, Adagrad and He with added Dropout
    add_dropout = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.HeNormal())
    ])
    add_dropout.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                        metrics=metrics)

    # NN with 3 Relu activated layers, Adagrad and He with added Dropout
    activity_reg = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.ActivityRegularization(l1=0.0001, l2=0.0001),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.ActivityRegularization(l1=0.0001, l2=0.0001),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.ActivityRegularization(l1=0.0001, l2=0.0001),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.HeNormal())
    ])
    activity_reg.compile(loss='categorical_crossentropy',
                         optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                         metrics=metrics)

    # NN with 3 Relu activated layers, SGD and He
    early_stopping = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax',
                              kernel_initializer=tf.keras.initializers.HeNormal())
    ])
    early_stopping.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                           metrics=metrics)

    my_own_final = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(900, activation='elu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.ActivityRegularization(l1=0.001, l2=0.001),
        tf.keras.layers.Dense(100,  kernel_initializer=tf.keras.initializers.HeNormal()),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(labels.shape[0], activation='softmax')
    ])
    my_own_final.compile(loss='categorical_crossentropy',
                         optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                         metrics=metrics)

    models = {'start model(only softmax)': start,
              'tanh_SGD_Glorot': only_tanh_MLP,
              'relu_SGD_He': only_ReLU_MLP,
              'Parametric_ReLu_upgrade': parametric_Relu,
              'Adagrad_upgrade': adagrad,
              'Batch-normalization_upgrade': batch_normalized,
              'Dropout_upgrade': add_dropout,
              'Activity_regularization': activity_reg,
              'Early_stopping': early_stopping,
              'Final_byZinoviy': my_own_final}

    result_dic = {'name': [],
                  'time': [],
                  'loss': [],
                  'test_acc': [],
                  'test_f1': []}

    for name in models:
        print(name)
        model = models[name]
        logdir = os.path.join("logs", name)
        modeldir = os.path.join("lab2_models", name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        callback = [tensorboard_callback]
        if (name == 'Early_stopping'):
            callback.append(tf.keras.callbacks.EarlyStopping(patience=2))
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
        result_dic['test_acc'].append(history.history['val_accuracy'][-1])
        result_dic['test_f1'].append(history.history['val_f1_score'][-1])
        print(history.history['val_accuracy'][-1])
        tf.keras.backend.clear_session()

    result = pd.DataFrame(result_dic)
    result.to_csv('logs/result.csv')
    print(result)
    return result
