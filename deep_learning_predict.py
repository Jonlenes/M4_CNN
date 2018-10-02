
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    from keras.models import load_model
    from keras.applications.resnet50 import ResNet50, preprocess_input
    from time import time
    from load_dataset import load_test_tensors
    from sklearn.metrics import accuracy_score, classification_report

    import numpy as np
    import keras.applications as app

preprecess = [app.resnet50.preprocess_input, 
            app.inception_v3.preprocess_input,
            app.xception.preprocess_input]


if __name__ == "__main__":
    """ 
    Realiza a predição sobre um rede treinada
    """

    for i in [0, 1, 2]:
        model_path = "saved_models/NEW/weights.best." + str(i) + ".hdf5"

        print("Carregando o modelo treinado....")
        t0 = time()
        model = load_model(model_path)
        t1 = time()
        print('Loaded in:', t1 - t0)

        test_tensors, targets_test = load_test_tensors()
        test_tensors = preprecess[i](test_tensors)

        ### Calculate classification accuracy on the test dataset.
        pred = np.argmax(model.predict(test_tensors), axis=1)
        print('Accuracy: %.4f%%' % (100 * accuracy_score(targets_test, pred)))

    # Report test accuracy
    # test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(targets_valid, axis=1))/len(predictions)
    


    # print("Starting the predict....")
    # pred = model.predict(valid_tensors)
    # pred = np.argmax(pred, axis=1)
    # print("Accuracy_score:", accuracy_score(targets_valid, list(pred)))
    # print(classification_report(targets_valid.ravel(), pred.ravel()))


    # print("Model.evaluate:", model.evaluate(valid_tensors, targets_valid, verbose=0)) 
    # pred_classes = model.predict_classes(valid_tensors)
    # print("Accuracy_score:", accuracy_score(targets_valid.ravel(), pred_classes.ravel()))
    ### Calculate classification accuracy on the test dataset.
    # Report test accuracy
    # test_accuracy = 100*np.sum(np.array(predictions)==targets_valid)/len(predictions)
    # print('Test accuracy: %.4f%%' % test_accuracy)
            

