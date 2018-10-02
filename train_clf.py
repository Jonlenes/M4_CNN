from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import scale
from time import time
from util import load_csv_file, path

import pandas
import os
import numpy as np


def load_base_dataset():
    """
    Carrega o dataset com o features e targets para ser executado nos clfs
    """

    # train = load_csv_file("train_aug.csv").values
    # train_target = load_csv_file("train_target_aug.csv").values
    # val = load_csv_file("val_aug.csv").values
    # val_target = load_csv_file("val_target_aug.csv").values
    
    # train = load_csv_file("dataset_train2.csv").values
    # train_target = load_csv_file("train_target.csv").values
    # val = load_csv_file("dataset_val2.csv").values
    # val_target = load_csv_file("val_target.csv").values

    # train = load_csv_file("dataset_meta_train.csv").values
    # train_target = load_csv_file("dataset_meta_target_train.csv").values
    # val = load_csv_file("dataset_meta_test.csv").values
    # val_target = load_csv_file("dataset_meta_target_test.csv").values

    #x_train = np.concatenate((x_train, pred_train), axis=1)
    #x_test = np.concatenate((x_test, pred_test), axis=1)

    train = []
    val = []
    train_target = []
    val_target = []

    for i in range(3):
        for j in range(10):
            
            x1 = load_csv_file("dataset_meta_train_" + str(i) + "_" + str(j) + ".cvs").values
            x2 = load_csv_file("dataset_meta_test_" + str(i) + "_" + str(j) + ".cvs").values
            
            y1 = np.argmax(load_csv_file("dataset_meta_target_train_" + str(i) + "_" + str(j) + ".cvs").values, axis=1).reshape(8300, 1) 
            y2 = load_csv_file("dataset_meta_target_test_" + str(i) + "_" + str(j) + ".cvs").values

            x1 = np.hstack((y1, x1))
            x2 = np.hstack((y2, x2))

            df1 = pandas.DataFrame(x1)
            df2 = pandas.DataFrame(x2)

            df1 = df1.sort_values(by=df1.columns.tolist())
            df2 = df2.sort_values(by=df2.columns.tolist())

            x1 = df1.values[:, 1:]
            x2 = df2.values[:, 1:]

            if len(train) == 0:
                train = x1
                val = x2
                train_target = df1.values[:, :1]
                val_target = df2.values[:, :1]
            else:
                train = np.concatenate((train, x1), axis=1)
                val = np.concatenate((val, x2), axis=1)

    return train, train_target, val, val_target

def rescaling(x):
    """
    feature scaling
    """
    for j in range(x.shape[1]):
        arr = x[:, j]
        min_val = np.min(arr)
        max_val = np.max(arr)
        x[:, j] = (x[:, j] - min_val) / (max_val - min_val)
    return x


def mean_normalisation(x):
    """
    feature scaling
    """
    for j in range(x.shape[1]):
        arr = x[:, j]
        min_val = np.min(arr)
        max_val = np.max(arr)
        mean = np.mean(arr)
        x[:, j] = (x[:, j] - mean) / (max_val - min_val)
    return x


# Nome dos classificadores
names = ["Linear SVM", "Random Forest", "Neural Net", "Logistic Regression"]

classifiers = [
    # KNeighborsClassifier(n_jobs=-1),
    SVC(kernel="linear", C=0.025),
    # SVC(),
    # DecisionTreeClassifier(),
    RandomForestClassifier(n_jobs=-1, n_estimators=1000),
    MLPClassifier(),
    # GaussianNB(),
    LogisticRegression()
    ]


print("Starting load datas....")
X_train, y_train, X_val, y_val = load_base_dataset()

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

print("Starting the scale...")
X_train = mean_normalisation(X_train)
X_val = mean_normalisation(X_val)

print("Starting the training...")

# rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100000)
# rfc.fit(X_train, y_train.ravel())
# print(rfc.score(X_val, y_val.ravel()))

# iterate over classifiers
for name, clf in zip(names, classifiers):   
    t = time()
    clf.fit(X_train, y_train.ravel())
    pred = clf.predict(X_val)
    score = clf.score(X_val, y_val.ravel())
    
    # print(classification_report(y_val, pred))
    print(name + ":", "accuracy:", accuracy_score(y_val, pred), "Time:", time() - t, "f1:", f1_score(y_val, pred, average='weighted'))