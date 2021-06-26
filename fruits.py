from sklearn.datasets import load_files
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
from data_preparation import *
from lab2 import lab2_run_models
from lab4 import lab4_run_models, lab_4_part1
from  sklearn.model_selection import train_test_split
from time import time

x_train, y_train, labels = load_prapared_dataset(TRAIN_PATH_DEMO)
x_test, y_test, _ = load_prapared_dataset(VAL_PATH_DEMO)

def boarder():
    for i  in range(3): print('=='*40)
'''
start = time()
lab2_result = lab2_run_models(x_train, y_train, labels, x_test, y_test)
print(lab2_result)
print(time()-start)
boarder()
boarder()
start = time()
x_val, x_t, y_val, y_t = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
lab3_result = lab4_run_models(x_train, y_train, labels, x_t, y_t, x_val, y_val)
print(lab3_result)
print(time()-start)

'''
boarder()
start = time()
x_val, x_t, y_val, y_t = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

lab_4_part1(x_train, y_train, labels, x_t, y_t, x_val, y_val, first=True)

lab_4_part1(x_train, y_train, labels, x_t, y_t, x_val, y_val)
print(time()-start)