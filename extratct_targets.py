import pandas as pd
from pandas import DataFrame as df
from util import get_files_names

""" 
Extrai as classes pelo nomes dos arquivos e salva em um arquivo csv
"""


def get_dataset(type):
    files = get_files_names(type)
    labels = [int(file.replace("\\", "/").split("/")[-1][:2]) for file in files]
    return labels


if __name__ == '__main__':
    train_targets = get_dataset("train")
    valid_targets = get_dataset("val")

    df(train_targets).to_csv(path + "train_target.csv", mode='a', header=True, index=False)
    df(valid_targets).to_csv(path + "val_target.csv", mode='a', header=True, index=False)
