import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    from keras.models import load_model
    from time import time
    from load_dataset import load_test_tensors_aug, load_dataset_tensors_aug
    from pandas import DataFrame as df
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import scale

    import numpy as np
    import keras.applications as app

preprecess = [app.resnet50.preprocess_input,
              app.inception_v3.preprocess_input,
              app.xception.preprocess_input]

if __name__ == "__main__":
    """ 
    Meta learning com data aug - Le report para entender isso melhor
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in [0, 1, 2]:
        model_path = "saved_models/75_epochs/weights.best." + str(i) + ".hdf5"

        print("Carregando o modelo treinado....")
        t0 = time()
        model = load_model(model_path)
        t1 = time()
        print('Loaded in:', t1 - t0)

        # train_generator, _ = load_dataset_tensors_aug(8300, preprecess[i])
        test_generator = load_test_tensors_aug(5420, preprecess[i])
        
        j = 0
        for (x, y) in test_generator:
            pred_train = model.predict(x)
            df(pred_train).to_csv("dataset_meta_test_" + str(i) + "_" + str(j) + ".cvs", mode='a', header=True, index=False)
            df(y).to_csv("dataset_meta_target_test_" + str(i) + "_" + str(j) + ".cvs", mode='a', header=True, index=False)
            j += 1
            if j >= 10:
                break
        # train_tensors, targets_train = load_train_tensors()
        # test_tensors, targets_test = load_test_tensors()

        # train_tensors = preprecess[i](train_tensors)
        # test_tensors = preprecess[i](test_tensors)

        # pred_train = model.predict(train_tensors)
        # pred_test = model.predict(test_tensors)

        # if len(x_train) == 0:
        #     x_train = pred_train
        #     y_train = targets_train
        #     x_test = pred_test
        #     y_test = targets_test
        # else:
        #     x_train = np.concatenate((x_train, pred_train), axis=1)
        #     x_test = np.concatenate((x_test, pred_test), axis=1)

    # df(x_train).to_csv("dataset_meta_train.cvs", mode='a', header=True, index=False)
    # df(y_train).to_csv("dataset_meta_target_train.cvs", mode='a', header=True, index=False)

    # df(x_test).to_csv("dataset_meta_test.cvs", mode='a', header=True, index=False)
    # df(y_test).to_csv("dataset_meta_target_test.cvs", mode='a', header=True, index=False)

    
    '''names = ["Linear SVM", "Random Forest", "Neural Net", "Logistic Regression"]

    classifiers = [
        SVC(kernel="linear", C=0.025),
        RandomForestClassifier(n_jobs=-1, n_estimators=1000),
        MLPClassifier(),
        LogisticRegression(n_jobs=-1)]

    print("Starting the scale...")
    X_train = scale(x_train)
    X_test = scale(x_test)

    print("Starting the training...")

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        t = time()
        clf.fit(X_train, y_train.ravel())
        score = clf.score(x_test, y_test.ravel())
        print(name + ":", score, "Time:", time() - t)'''
