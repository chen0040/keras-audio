from keras_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found


def main():
    data_dir_path = '../very_large_data/gtzan'
    download_gtzan_genres_if_not_found(data_dir_path)


if __name__ == '__main__':
    main()
