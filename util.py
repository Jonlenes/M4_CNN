from keras.preprocessing.image import load_img, img_to_array
from joblib import Parallel, delayed                            

import multiprocessing
import numpy as np
import glob
import os
import pandas

path = ""
if os.name == 'nt':
    path = "E:/OneDrive/MO444/MO444_dogs/"
else:
    path = "/home/jonlenes/Desktop/MO444_dogs/"


def get_files_names(subfolder):
    """ 
    Retonar todas os nomes dos arquivos jpg em uma diretorio
    """
    return glob.glob(path + subfolder + "/" + "*.jpg")


def load_csv_file(file_name):
    """
    Carrega um csv
    """
    return pandas.read_csv(path + file_name)


def path_to_tensor(img_path, i):
    """
    Converte uma imagem para tensor
    """
    #indice da imagem que está sendo processada
    print("Processando %d" % i)
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """
    Converte um conjunto de imagens para tensor utilizando todas as CPUs paralelamente
    """
    num_cores = multiprocessing.cpu_count()
    print("Processando com %d cores..." % num_cores)

    list_of_tensors = Parallel(n_jobs=num_cores)(delayed(path_to_tensor)(img_paths[i], i) for i in range(len(img_paths)))
    return np.vstack(list_of_tensors) 
