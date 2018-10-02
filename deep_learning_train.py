import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    from load_dataset import get_dataset
    from keras.applications.resnet50 import ResNet50
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.xception import Xception
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from load_dataset import load_dataset_tensors
    from keras.optimizers import SGD
    from keras.applications.imagenet_utils import preprocess_input

    import numpy as np
    import keras.applications as app


def load_basic_model():
    """
    Modelo básico contruido neste trabalho
    """
    
    model = Sequential()

    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(input_shape=(224, 224, 3), filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())

    model.add(Dense(83, activation='softmax'))

    model.summary()

    return model


def pre_trained_model(index):
    """
    Retorno modelos pre-treinados com suas funções de pre-processamento.
    """

    if index == 0:
        return ["ResNet50", ResNet50(weights='imagenet', include_top=False), app.resnet50.preprocess_input]
    elif index == 1:
        return ["InceptionV3", InceptionV3(weights='imagenet', include_top=False), app.inception_v3.preprocess_input]
    return ["Xception", Xception(weights='imagenet', include_top=False), app.xception.preprocess_input]


def finetuned_model(x_train, y_train, valid_tensors, targets_valid, base_model, epochs, batch_size, index_model):
    """
    Realiza o fine tune dos modelos
    """

    # get layers and add average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add fully-connected layer
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu', name='fc-2')(x)
    # a softmax layer for 4 classes
    out = Dense(83, activation='softmax', name='output_layer')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in base_model.layers:
        layer.trainable = False

    # update the weight that are added - Treinando uma vez com as camadas congeladas
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(x_train, y_train)

    # choose the layers which are updated by training
    layer_num = len(model.layers)
    perct_freezed = 0.75

    for layer in model.layers[:int(layer_num * perct_freezed)]:
        layer.trainable = False

    for layer in model.layers[int(layer_num * perct_freezed):]:
        layer.trainable = True

    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="saved_models/weights.best." + str(index_model) + ".hdf5", verbose=1, save_best_only=True)
    model.fit(x_train, y_train, validation_data=(valid_tensors, targets_valid), epochs=epochs, callbacks=[checkpointer], batch_size=batch_size)

    return model


if __name__ == "__main__":

    basic_model = False
    data_augmentation = False
    epochs = 5
    batch_size = 32 
    # n_examples = train_tensors.shape[0]
    
    print(epochs, batch_size)
    
    if basic_model:
        
        print("Montando o modelo basico..")
        model = load_basic_model()

        print("Compilando o modelo...")
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        print("Treinando....")
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
        
        print("Ajuste o modelo!")
        model.fit(train_tensors, targets_train, validation_data=(valid_tensors, targets_valid),
               epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

        print("Predição da validação....")
        # Load the Model with the Best Validation Loss
        model.load_weights('saved_models/weights.best.from_scratch.hdf5')

        # get index of predicted dog breed for each image in test set - vou trocar test por validação aqui
        dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in valid_tensors]

        # report test accuracy
        test_acc = 100*np.sum(np.array(dog_breed_predictions) == np.argmax(targets_valid, axis=1))/len(dog_breed_predictions)
        print('Test accuracy: %.4f%%' % float(test_acc))

    else:
        for i in [0, 1, 2]:
            name, model, preprocess = pre_trained_model(i)
            print(name)

            train_tensors, targets_train, valid_tensors, targets_valid = load_dataset_tensors()
            train_tensors = preprocess(train_tensors)
            valid_tensors = preprocess(valid_tensors)

            tried_model = finetuned_model(train_tensors, targets_train, valid_tensors, targets_valid, model, epochs, batch_size, i)

            ### Load the model weights with the best validation loss.
            # finetuned_model.load_weights("saved_models/weights.best.ResNet50.hdf5")
            
            # get index of predicted dog breed for each image in test set
            predictions = tried_model.predict(valid_tensors)
            labels = np.argmax(predictions, axis=1)
            print(predictions.shape, labels.shape)

            # report test accuracy
            test_accuracy = 100 * np.sum(np.array(labels)==np.argmax(targets_valid, axis=1))/len(labels)
            print('Test accuracy: %.4f%%' % test_accuracy)

            del name, model, preprocess, tried_model, predictions, labels, test_accuracy, train_tensors, targets_train, valid_tensors, targets_valid
        

