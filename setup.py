from setuptools import setup

try:
    from pypandoc import convert

    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
    name='digit_sequence_recognize',
    version='1.0.0',
    author='wang zhaoxu',
    author_email='wangzhaoxu1985@gmail.com',

    packages=[
        'digit_sequence_recognize',
        'digit_sequence_recognize.digit_sequence_generator',
        'digit_sequence_recognize.train_model',
        'digit_sequence_recognize.utilities',
    ],

    url='',
    license='LICENSE.txt',

    description='ai library for sound data',
    long_description=read_md('README.md'),

    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'bcolz',
        'Pillow',
    ],

    classifiers=[

        'Programming Language :: Python',
        'Topic :: Sound'
    ],
)
