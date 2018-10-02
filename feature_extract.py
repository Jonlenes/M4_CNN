from load_dataset import get_dataset, path
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from pandas import DataFrame as df
from joblib import Parallel, delayed
from util import paths_to_tensor                            

import numpy as np
import multiprocessing


if __name__ == '__main__':
    """
    Realiza a extração de features das imagens utilizando a ResNet50 pretreinada (paralela)
    """

    model = ResNet50(include_top=False, weights='imagenet')

    print("Carregando os dados...")
    train_files, train_targets = get_dataset("train")
    valid_files, valid_targets = get_dataset("val")

    files_names = ["dataset_train3.csv", "dataset_val3.csv"]
    block_size = 500

    # 0 - train, 1 - validação
    for i_file in [0, 1]:
        
        files = [train_files, valid_files][i_file]
        count = int(np.ceil(len(files) / block_size))

        for i in range(count):
            print("{0} lotte: {1}/{2}".format(files_names[i_file], i, count))
            
            begin = i*block_size
            end = begin + block_size
            if end > len(files):
                end = len(files)
            
            # Processa um lote de imagens paralelamente
            tensors = paths_to_tensor(files[begin:end])
            tensors = preprocess_input(tensors)
            pred = model.predict(tensors).reshape(end - begin, 2048)

            # Adiciona essas imagens ao csv
            df(pred).to_csv(path + files_names[i_file], mode='a', header=(i==10), index=False)



