"""
the module learning a model for vector representations of words
"""
import os
import string
import gensim
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_texts_for_dataset(file, dir_name):
    """
    function to load text data for training the network.
    on entry accepts:
        file --------- str, file name to download
        dir_name ----- str, directory where the file is located
    returns to output:
        doc ---------- str, read and translated to bottom case text file
    """

    file_name = os.path.join(dir_name, file)
    with open(file_name, 'r', encoding='utf-8') as file_read:
        doc = file_read.read()

    return doc


def tokenize_ru(file_text):
    """
    function of training data to train the word2vec model
    on entry accepts:
        file_text ----- str, data read from file
    returns to output:
        tokens -------- list the list of tokens
    """

    tokens = word_tokenize(file_text)
    tokens = [i for i in tokens if (i not in string.punctuation)]
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—',
                       '–', 'к', 'на', '...', '@'])
    tokens = [i for i in tokens if (i not in stop_words)]
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens


def modelw2v_save(dir_save, model_w2v_name, model):
    """
    feature save the model to disk w2v
    on entry accepts:
        dir_save -------- str, directory to save w2v model
        model_w2v_name -- str, the name of the model to save
    """

    path_modelw2v_save = os.path.join(dir_save, model_w2v_name)
    print('сохранение модели в {}'.format(path_modelw2v_save))
    model.save(path_modelw2v_save)


def handler(file, dir_name, dir_save, model_w2v_name):
    """
    function loads the text from the disk, converts it into a sequence
    trains and saves the word2vec model.
    on entry receives:
        file ------------ str, file name with text data for training
        dir_name -------- str, directory with text data
        dir_save -------- str, directory to save the model
        model_w2v_name -- str, the name of the model to save
    """

    print('загрузка текста')
    text = load_texts_for_dataset(file, dir_name)
    sentences = [tokenize_ru(sent) for sent in sent_tokenize(text, 'russian')]

    model = gensim.models.Word2Vec(sentences,
                                   size=200, window=5, min_count=10, workers=4)
    model.init_sims(replace=True)

    modelw2v_save(dir_save, model_w2v_name, model)


def main():
    """
    the start function of the learning process
    """
    file = 'result.txt'
    dir_name = r'..\data\w2v'
    dir_save = r'..\models'
    model_w2v_name = 'word2vec_gensim.bin'
    handler(file, dir_name, dir_save, model_w2v_name)


if __name__ == '__main__':
    main()
