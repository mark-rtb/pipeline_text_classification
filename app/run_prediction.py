"""
Loading module pridorozhnoy network, downloading and word processing
and an example of performing a class probability prediction.
"""
import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from keras.models import model_from_json
from rutermextract import TermExtractor
from preprocessing import text_preprocessing

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
        doc = text_preprocessing(file_read.read())

    return doc


def text_to_vector(text, model_w2v, maxlen):
    """
    Function to convert a text file to an array where words are encoded
    word2vec method.
    on entry receives:
        text ------------- str, transcoding document
        model_w2v -------- model, the model which is coding
        maxlen ----------- int, the maximum length of the sequence to which
        are all the rows of the array.
    returns to output:
        arr_word_vector -- np.array an array of strings encoded w2v
    """

    list_word_vector = []
    for word in text:
        try:
            vector = model_w2v.wv[word]
            list_word_vector.append(vector)
        except:
            vector = model_w2v.wv['0']
            list_word_vector.append(vector)
    if len(list_word_vector) >= maxlen:
        list_word_vector = list_word_vector[:maxlen]
    else:
        difference = maxlen - len(list_word_vector)
        for _ in range(difference):
            list_word_vector.append(model_w2v.wv['0'])
            list_word_vector = list_word_vector[:maxlen]

    arr_word_vector = np.array(list_word_vector)
    return arr_word_vector


def load_model(dir_name, file_name_model, file_name_weight):
    """
    download function pridorozhnoy model for making predictions
    at the entrance:
        dir_name ---------- str, the directory where the model is stored
        file_name_model --- str, the name of the classification model
        file_name_weight -- str, the name of the file with scale models
    at the output:
        model ------------- model, model for prediction execution
    """

    name_model = os.path.join(dir_name, file_name_model)
    name_weight = os.path.join(dir_name, file_name_weight)

    with open(name_model, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(name_weight)

    return model


def load_dickt(dir_name, file_name_dickt):
    """
    download dictionary of conformity of category names labels
    on entry accepts:
        dir_name -------- str, dictionary download directory
        file_name_dickt -- str, the name of the dictionary category names
    returns to output:
        name_to_lable --- dict, a dictionary matching of the
        category names to the labels
    """

    path_dickt = os.path.join(dir_name, file_name_dickt)
    file_w = os.path.join(path_dickt)
    with open(file_w, 'rb') as file_word:
        name_to_lable = pickle.load(file_word)

    return name_to_lable


def load_w2v_model(dir_name, model_w2v_name):
    """
    model loading function to convert words into vectors
    on entry accepts:
        dir_name -------- str, w2v model download directory
        model_w2v_name -- str, the name of the model w2v
    returns to output:
        model_w2v ------- model, loaded w2v model
    """

    path_model_w2v = os.path.join(dir_name, model_w2v_name)
    model_w2v = Word2Vec.load(path_model_w2v)

    return model_w2v


def theme(text):
    """
    function of extracting the key themes from the text
    on entry accepts:
        text ------------ str, text to extract the topic
    returns to output:
        theme ----------- str, three main topics from the text
    """

    term_extractor = TermExtractor()
    list_theme = []
    for term in term_extractor(text):
        list_theme.append(term.normalized)
    theme_text = ' | '.join(list_theme[:3])
    return theme_text



def predict_files(file_name_dickt, dir_name, file_name_model, file_name_weight,
                  dir_test, model_w2v_name, file_name_model_exec,
                  file_name_weight_exec, file_name_dickt_ex):
    """
    function of loading necessary modules, loading data for prediction
    and output of predicted classes.
    on entry accepts:
        file_name_dickt ----- str, the name of the dictionary category names
        dir_name ------------ str, dictionary and model download directory
        file_name_model ----- str, the name of the classification model
        file_name_weight ---- str, the name of the file with scale models
        dir_test ------------ str, directory with files for testing
        model_w2v_name ------ str, loaded w2v model
    """

    model_w2v = load_w2v_model(dir_name, model_w2v_name)
    name_to_lable = load_dickt(dir_name, file_name_dickt)
    name_executor = load_dickt(dir_name, file_name_dickt_ex)
    model = load_model(dir_name, file_name_model, file_name_weight)
    model_exec = load_model(dir_name, file_name_model_exec, file_name_weight_exec)
    maxlen = 500

    list_file_test = os.listdir(dir_test)
    for file_pred in list_file_test:
        text_pred = load_texts_for_dataset(file_pred, dir_test)
        vector_pred = text_to_vector(text_pred, model_w2v, maxlen)
        vector_pred = np.array([vector_pred])
        doc_type = model.predict(vector_pred)
        doc_exec = model_exec.predict(vector_pred)
        for name, val in enumerate(name_to_lable):
            if name == np.argmax(doc_type):
                list_ansver = [val, np.max(doc_type)]
        print('Тема обращения: {}'.format(theme(text_pred)))
        print('Файл отнесен к категории: {},  с вероятностью {}%'
              .format(list_ansver[0], int(round(list_ansver[1]*100, 0))))

        for name, val in enumerate(name_executor):
            if name == np.argmax(doc_exec):
                list_ansver = [val, np.max(doc_exec)]
        print('Предпологаемый исполнитель: {},  с вероятностью {}%\n\n'
              .format(list_ansver[0], int(round(list_ansver[1]*100, 0))))


def main():
    """
    test run function
    """
    file_name_dickt = 'dict_labels.pickle'
    file_name_dickt_ex = 'dict_labels_executor.pickle'
    dir_name = r'..\models'
    file_name_model = 'conv1D_W2V.json'
    file_name_weight = 'conv1D_W2V.h5'
    dir_test = r'..\data\test'
    model_w2v_name = 'word2vec_gensim.bin'
    file_name_model_exec = 'conv1D_W2V_executor.json'
    file_name_weight_exec = 'conv1D_W2V_executor.h5'


    predict_files(file_name_dickt, dir_name, file_name_model, file_name_weight,
                  dir_test, model_w2v_name, file_name_model_exec,
                  file_name_weight_exec, file_name_dickt_ex)


if __name__ == '__main__':
    main()
