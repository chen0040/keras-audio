from lru import LRU

import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras_audio.library.resnets_utils import *
from keras_audio.library.resnets_utils import convert_to_one_hot, load_dataset
from keras_audio.library.utility.audio_utils import compute_melgram


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('elu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('elu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)

    return X


def identity_block_test():
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(F1, (1, 1), padding='valid', strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('elu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), padding='same', strides=(1, 1), name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('elu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), padding='valid', strides=(1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), padding='valid', strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)

    return X


def convolutional_block_test():
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))


def resnet_50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('elu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # X = Dropout(rate=0.25)(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)

    # X = Dense(units=512)(X)
    # X = Activation('elu')(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def resnet_50_test():
    model = resnet_50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    dataset_dir_path = '../training/resnet_datasets'

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(dataset_dir_path)

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model.fit(X_train, Y_train, epochs=2, batch_size=32)

    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))


class ResNet50AudioClassifier(object):
    model_name = 'resnet50'

    def __init__(self):
        self.cache = LRU(400)
        self.input_shape = None
        self.nb_classes = None
        self.model = None
        self.config = None

    def create_model(self):
        self.model = resnet_50(input_shape=self.input_shape, classes=self.nb_classes)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.model.summary())

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, ResNet50AudioClassifier.model_name + '-config.npy')

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, ResNet50AudioClassifier.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, ResNet50AudioClassifier.model_name + '-weights.h5')

    def load_model(self, model_dir_path):
        config_file_path = ResNet50AudioClassifier.get_config_file_path(model_dir_path)
        weight_file_path = ResNet50AudioClassifier.get_weight_file_path(model_dir_path)
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
            # mg = (mg + 100) / 200  # scale the values
            self.cache[audio_path] = mg
            return mg

    def generate_batch(self, audio_paths, labels, batch_size):
        num_batches = len(audio_paths) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size

                X = np.zeros(shape=(batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
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

        config_file_path = ResNet50AudioClassifier.get_config_file_path(model_dir_path)
        weight_file_path = ResNet50AudioClassifier.get_weight_file_path(model_dir_path)
        architecture_file_path = ResNet50AudioClassifier.get_architecture_file_path(model_dir_path)

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

        np.save(os.path.join(model_dir_path, ResNet50AudioClassifier.model_name + '-history.npy'), history.history)
        return history

    def predict(self, audio_path):
        mg = compute_melgram(audio_path)
        mg = np.expand_dims(mg, axis=0)
        return self.model.predict(mg)[0]

    def predict_class(self, audio_path):
        predicted = self.predict(audio_path)
        return np.argmax(predicted)


def main():
    identity_block_test()
    convolutional_block_test()
    resnet_50_test()


if __name__ == '__main__':
    main()
