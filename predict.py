from load_dataset import get_dataset
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from PIL import ImageFile
from joblib import Parallel, delayed
from util import paths_to_tensor                           

import numpy as np
import multiprocessing


if __name__ == "__main__":
    print("Carregando os dados...")
    valid_files, valid_targets = get_dataset("val")

    # pre-process the data for Keras
    print("Convertendo para tensor...")
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255.0

    model = Sequential()
    model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    model.add(Flatten(name='flatten'))
    model.add(Dense(83, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    ### Load the model weights with the best validation loss.
    model.load_weights('saved_models/weights.best.ResNet50.hdf5')

    ### Calculate classification accuracy on the test dataset.
    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in valid_tensors]

    # Report test accuracy
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(valid_targets, axis=1))/len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)