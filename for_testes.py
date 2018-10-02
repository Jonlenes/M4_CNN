import numpy as np
import os
import pandas

from util import load_csv_file, path

train = load_csv_file("train_target_aug.csv").values
array = np.argmax(train, axis=1)
pandas.DataFrame(array).to_csv(path + "train_target_aug.csv", header=True, index=False)