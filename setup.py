"""
Created on Tue Jul 30 15:51:29 2019

@author: Admin
"""


from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='automatic_classification_citizens_appeals_Ryazan region',
    version='1.0.0',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
#    Platform='python3.6.8',
#    Author='mark-rtb@yandex.ru',
    install_requires=[
        'numpy==1.17.0',
        'gensim==3.7.1',
        'nltk==3.4.5',
        'pandas==0.24.0',
        'pymorphy2==0.8',
        'scikit-learn==0.20.2',
        'tensorflow==2.9.3',
        'Keras==2.2.4',
        'h5py==2.8.0',
        'matplotlib==3.0.2',
        'rutermextract==0.3',
        'googletrans==2.4.0',
    ]
)
