from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from util import paths_to_tensor, get_files_names, path

import numpy as np
import os

def get_dataset(type, simples=False):
    """ 
    Retona a lista de path das imagens e seus labels
    """
    files = get_files_names(type)
    labels = [int(file.replace("\\", "/").split("/")[-1][:2]) for file in files]
    if not simples:
        labels = np_utils.to_categorical(np.array(labels), 83)  

    return files, labels


def load_dataset_tensors():
    """ 
    Retonar os tensors para serem treinados com o Keras
    """
    print("Carregando os dados...")
    train_files, targets_train = get_dataset("train")
    valid_files, targets_valid = get_dataset("val")
    
    # pre-process the data for Keras
    print("Convertendo para tensor...")
    train_tensors = paths_to_tensor(train_files) 
    valid_tensors = paths_to_tensor(valid_files)

    return train_tensors, targets_train, valid_tensors, targets_valid


def load_train_tensors():
    """ 
    Retonar os tensors para serem treinados com o Keras (Só treinamento)
    """
    print("Carregando os dados...")
    train_files, targets_train = get_dataset("train", True)

    # pre-process the data for Keras
    print("Convertendo para tensor...")
    train_tensors = paths_to_tensor(train_files)

    return train_tensors, targets_train


def load_test_tensors(val=False):
    """ 
    Retonar os tensors de test
    """
    
    print("Carregando os dados...")
    if val:
        test_files, targets_test = get_dataset("val", True)
    else:
        test_files, targets_test = get_dataset("test", True)
    
    # pre-process the data for Keras
    print("Convertendo para tensor...")
    test_tensors = paths_to_tensor(test_files)

    return test_tensors, targets_test


def load_dataset_tensors_aug(batch_size, preprocess_function):
    """ 
    Retorna o train_generator e test_generator com data aug
    """

    print("Incializando o pre data Augmentation...")
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_function,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    #rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest')
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_function, 
                                    #rescale=1. / 255
                                    )
    
    train_tensors, targets_train, valid_tensors, targets_valid = load_dataset_tensors()

    train_datagen.fit(train_tensors)
    test_datagen.fit(valid_tensors)

    train_generator = train_datagen.flow(train_tensors, targets_train, batch_size=batch_size)
    test_generator = test_datagen.flow(valid_tensors, targets_valid, batch_size=batch_size)

    return train_generator, test_generator


def load_test_tensors_aug(batch_size, preprocess_function):
    """ 
    Retorna o test_generator com data aug
    """

    print("Incializando o pre data Augmentation...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_function,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    #rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest')
    
    test_tensors, targets_test = load_test_tensors()
    test_generator = test_datagen.flow(test_tensors, targets_test, batch_size=batch_size)
    return test_generator


def set_class_dataset_test():
    """ 
    Renomeia o dataset de test para facilitar o carregamento dos dados
    """
    
    text_file = open(path + "MO444_dogs_test.txt", "r")
    lines = text_file.readlines()
    for line in lines:
        class_dog = int(line.split()[2])
        file_name = line.split()[1].split("/")[-1]
        os.rename(path + "test/" + file_name, path + "test/" + str(class_dog).zfill(2) + "_" + file_name)


if __name__ == '__main__':

    train_files, train_targets = get_dataset("train",)
    valid_files, valid_targets = get_dataset("val")
    test_files, test_targets = get_dataset("test")

    # print('There are %d total dog categories.' % len(set(train_targets)))
    print('There are %d total dog images.' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.' % len(test_files))
