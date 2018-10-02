from load_dataset import get_dataset, path
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from pandas import DataFrame as df
from joblib import Parallel, delayed
from util import paths_to_tensor                            

import numpy as np
import multiprocessing


if __name__ == '__main__':
    """
    Realiza a extração de features das imagens utilizando a ResNet50 pretreinada (paralela) - data aug
    """
    model = ResNet50(include_top=False, weights='imagenet')

    print("Carregando os dados...")
    train_files, train_targets = get_dataset("train", 1, True)
    valid_files, valid_targets = get_dataset("val", 1, True)

    data_augmentation = True    

    files_names = [["train_aug.csv", "train_target_aug.csv"], ["val_aug.csv", "val_target_aug.csv"]]
    block_size = 500
    n_examples = [len(train_files), len(valid_files)]

    for i_file in [0]:
        files = [train_files, valid_files][i_file]
        target = [train_targets, valid_targets][i_file]
        
        print("Incializando o pre data Augmentation...")
        datagen_train = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                # rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
        tensors = paths_to_tensor(files)
        datagen_train.fit(tensors)
        
        j = 0
        for X_batch, y_batch in datagen_train.flow(tensors, target, batch_size=n_examples[i_file]):
            print("Gerando batch %d" % j)
            
            count = int(np.ceil(len(X_batch) / block_size))
            print(len(X_batch),  count)
            # for i in range(count):
            #    print("{0} lotte: {1}/{2}".format(files_names[i_file][0], i, count))
            #    begin = i*block_size
            #    end = begin + block_size
            #    if end > len(X_batch):
            #        end = len(X_batch)
            tensors = preprocess_input(X_batch)
            pred = model.predict(tensors).reshape(n_examples[i_file], 2048)
            df(pred).to_csv(path + files_names[i_file][0], mode='a', header=(j==0), index=False)
            df(y_batch).to_csv(path + files_names[i_file][1], mode='a', header=(j==0), index=False)
            
            j += 1
            if j == 10:
                break
