from keras_audio.library.cifar10 import Cifar10AudioClassifier


def main():

    classifier = Cifar10AudioClassifier()
    classifier.load_model(model_dir_path='./models')

    classifier.export_tensorflow_model(output_fld='./models/tensorflow_models/cifar10')


if __name__ == '__main__':
    main()
