from pydl_audio_encoders.library.utility.gtzan_loader import download_gtzan_genres


def main():
    data_dir_path = '../very_large_data/gtzan'
    download_gtzan_genres(data_dir_path)


if __name__ == '__main__':
    main()
