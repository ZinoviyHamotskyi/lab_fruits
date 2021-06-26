from sklearn.datasets import load_files
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
import warnings
warnings.filterwarnings("ignore")


VAL_PATH_DEMO = "Fruits_demo_4/Validation/"
TRAIN_PATH_DEMO = "Fruits_demo_4/Training/"
VAL_PATH = "Fruits_full/Validation/"
TRAIN_PATH = "Fruits_full/Validation/"


def load_dataset(data_path):
    data_loading = load_files(data_path)
    files_add = np.array(data_loading['filenames'])
    targets_fruits = np.array(data_loading['target'])
    target_labels_fruits = np.array(data_loading['target_names'])
    return files_add,targets_fruits,target_labels_fruits

def convert_image_to_array_form(files):
    images_array=[]
    for file in files:
        images_array.append(img_to_array(load_img(file)))
    return images_array

def load_prapared_dataset(path):
    x, y, labels = load_dataset(path)
    print("dataset from " + path + " was successfully loaded")
    x_np = np.array(convert_image_to_array_form(x))
    x_normalized = x_np.astype('float32')/255
    y_categorical = np_utils.to_categorical(y, labels.shape[0])
    print("dataset was already prepared")
    return x_normalized, y_categorical, labels

def noise(array):
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

'''

x_test_pic, y_test,_ = load_dataset(VAL_PATH_DEMO)
print('test set has been uploaded')
x_train_pic, y_train,target_labels = load_dataset(TRAIN_PATH_DEMO)
print('train set has been uploaded')
print(x_test_pic[0])

n_classes = target_labels.shape[0]
y_train = np_utils.to_categorical(y_train,n_classes)
y_test = np_utils.to_categorical(y_test,n_classes)



x_train_un = np.array(convert_image_to_array_form(x_train_pic))
print('Training set shape : ',x_train_un.shape)

x_test_un = np.array(convert_image_to_array_form(x_test_pic))
print('Test set shape : ',x_test_un.shape)

x_train = x_train_un.astype('float32')/255
x_test = x_test_un.astype('float32')/255
'''