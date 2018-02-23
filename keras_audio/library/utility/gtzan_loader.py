import urllib.request
import os
import tarfile

from pydl_audio_encoders.library.utility.download_utils import reporthook


def download_gtzan_music_speech(data_dir_path):
    zip_file_path = data_dir_path + '/music_speech.tar.gz'

    if not os.path.exists(zip_file_path):
        url_link = 'http://opihi.cs.uvic.ca/sound/music_speech.tar.gz'
        print('gz model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_link, filename=zip_file_path,
                                   reporthook=reporthook)

    tar = tarfile.open(zip_file_path, "r:gz")
    tar.extractall(data_dir_path)
    tar.close()


def download_gtzan_genres(data_dir_path):
    zip_file_path = data_dir_path + '/genres.tar.gz'

    if not os.path.exists(zip_file_path):
        url_link = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
        print('gz model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_link, filename=zip_file_path,
                                   reporthook=reporthook)

    tar = tarfile.open(zip_file_path, "r:gz")
    tar.extractall(data_dir_path)
    tar.close()

