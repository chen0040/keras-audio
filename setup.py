from setuptools import setup

setup(
    name='keras_audio',
    packages=['keras_audio'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn',
        'nltk',
        'numpy',
        'h5py'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)