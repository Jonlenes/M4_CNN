import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    from keras.applications.imagenet_utils import preprocess_input
    from keras.applications.resnet50 import ResNet50
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.xception import Xception
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Dropout, Flatten, Dense
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from keras.optimizers import SGD
    from load_dataset import load_dataset_tensors_aug
    from time import time

    import numpy as np
    import keras.applications as app


def pre_trained_model(index):
    """
    Retorno modelos pre-treinados com suas funções de pre-processamento.
    """
    if index == 0:
        return ["ResNet50", ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)), app.resnet50.preprocess_input]
    elif index == 1:
        return ["InceptionV3", InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3)), app.inception_v3.preprocess_input]

    return ["Xception", Xception(weights='imagenet', include_top=False, input_shape=(224,224,3)), app.xception.preprocess_input]


def finetuned_model(train_generation, test_generation, base_model, epochs, batch_size, index_model):
    """
    Realiza o fine tune dos modelos com data aug
    """
    n_examples_train = 8300
    n_examples_val = 6022

    # get layers and add average pooling layer
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    
    # last_layer = base_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(x)

    # add fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # add fully-connected & dropout layers
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.25)(x)
    # x = Dense(256, activation='relu', name='fc-2')(x)
    # x = Dropout(0.5)(x)

    # add output layer
    predictions = Dense(83, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in base_model.layers:
        layer.trainable = True

    # update the weight that are added
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generation, validation_data=test_generation, epochs=10)


    # choose the layers which are updated by training
    layer_num = len(model.layers)
    print("Numero de camadas:", layer_num)
    perct_freezed = 0.9

    for i in range(layer_num):
        model.layers[i].trainable = (i < int(layer_num * perct_freezed))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="saved_models/weights.best.aug." + str(index_model) + ".hdf5", verbose=1, save_best_only=True)
    model.fit_generator(train_generation, 
                        validation_data=test_generation, 
                        steps_per_epoch=n_examples_train // batch_size, 
                        validation_steps=n_examples_val // batch_size,
                        epochs=epochs, 
                        callbacks=[checkpointer])

    return model


if __name__ == "__main__":

    epochs = 20
    batch_size = 16 
    
    print(epochs, batch_size)
    
    for i in [0, 1, 2]:
        name, model, preprocess = pre_trained_model(i)
        print(name)

        train_generation, test_generation = load_dataset_tensors_aug(batch_size, preprocess)
        tried_model = finetuned_model(train_generation, test_generation, model, epochs, batch_size, i)

        del name, model, preprocess, tried_model, train_generation, test_generation
