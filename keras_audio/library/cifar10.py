import numpy as np
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from keras_audio.library.resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import keras.backend as K
import tensorflow as tf
from lru import LRU

from keras_audio.library.resnets_utils import convert_to_one_hot, load_dataset
from keras_audio.library.utility.audio_utils import compute_melgram


def cifar10(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class Cifar10AudioClassifier(object):
    model_name = 'resnet50'

    def __init__(self):
        self.cache = LRU(1000)
        self.input_shape = None
        self.nb_classes = None
        self.model = None
        self.config = None

    def create_model(self):
        self.model = cifar10(input_shape=self.input_shape, classes=self.nb_classes)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.model.summary())

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-config.npy')

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-weights.h5')

    def load_model(self, model_dir_path):
        config_file_path = Cifar10AudioClassifier.get_config_file_path(model_dir_path)
        weight_file_path = Cifar10AudioClassifier.get_weight_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.input_shape = self.config['input_shape']
        self.nb_classes = self.config['nb_classes']
        self.create_model()
        self.model.load_weights(weight_file_path)

    def compute_melgram(self, audio_path):
        if audio_path in self.cache:
            return self.cache[audio_path]
        else:
            mg = compute_melgram(audio_path)
            self.cache[audio_path] = mg
            return mg

    def generate_batch(self, audio_paths, labels, batch_size):
        num_batches = len(audio_paths) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size

                X = np.zeros(shape=(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
                for i in range(start, end):
                    audio_path = audio_paths[i]
                    mg = compute_melgram(audio_path)
                    X[i - start, :, :, :] = mg
                yield X, labels[start:end]

    def fit(self, audio_path_label_pairs, model_dir_path, batch_size=None, epochs=None, test_size=None,
            random_state=None, input_shape=None, nb_classes=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if input_shape is None:
            input_shape = (96, 1366, 1)
        if nb_classes is None:
            nb_classes = 10

        config_file_path = Cifar10AudioClassifier.get_config_file_path(model_dir_path)
        weight_file_path = Cifar10AudioClassifier.get_weight_file_path(model_dir_path)
        architecture_file_path = Cifar10AudioClassifier.get_architecture_file_path(model_dir_path)

        self.input_shape = input_shape
        self.nb_classes = nb_classes

        self.config = dict()
        self.config['input_shape'] = input_shape
        self.config['nb_classes'] = nb_classes
        np.save(config_file_path, self.config)

        self.create_model()

        with open(architecture_file_path, 'wt') as file:
            file.write(self.model.to_json())

        checkpoint = ModelCheckpoint(weight_file_path)

        X = []
        Y = []

        for audio_path, label in audio_path_label_pairs:
            X.append(audio_path)
            Y.append(label)

        X = np.array(X)
        Y = np.array(Y)

        Y = np_utils.to_categorical(Y, self.nb_classes)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        np.save(os.path.join(model_dir_path, Cifar10AudioClassifier.model_name + '-history.npy'), history.history)
        return history
