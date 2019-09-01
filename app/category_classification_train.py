"""
Module training the neural network for classification
of applications type
"""

import os
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.wrappers import Bidirectional
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
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


def load_w2v_model(dir_name, model_w2v_name):
    """
    download function models for the encoding of words
    on entry accepts:
        dir_name -------- str, the directory where the model is located
        model_w2v_name -- str, the name of the model to load
    returns to output:
        model_W2V ------- model, loaded w2v model
    """

    path_model_w2v = os.path.join(dir_name, model_w2v_name)
    model_w2v = Word2Vec.load(path_model_w2v)

    return model_w2v


def create_df_from_dir(dir_texts, maxlen, model_w2v):
    """
    function of collecting the array for the training.
    During the execution are checked
    all directories in the transferred location, found text files a label is
    loaded, encoded, and created with the name of the directory from which
    they were uploaded.
    on entry accepts:
        dir_txt --------- str, directory where the data directories are located
        maxlen ---------- int, the maximum length of the sequence to which
        model_В2В ------- model, loaded B2B model
    returns to output:
        array_to_split -- np.array, an array with loaded texts
        list_table ------ list, list of class labels
        name_to_lable --- dict a dictionary mapping labels and the names
        of the classes
    """

    list_dir = os.listdir(dir_texts)
    name_to_lable = {}
    for i, name in  enumerate(list_dir):
        name_to_lable[name] = i
    list_vector = []
    list_lable = []
    for name in list_dir:
        dir_name = os.path.join(dir_texts, name)
        list_files = os.listdir(dir_name)
        for file in list_files:
            if file[-4:] == '.txt':
                text_to_v = load_texts_for_dataset(file, dir_name)
                vector = text_to_vector(text_to_v, model_w2v, maxlen)
                list_vector.append(vector)
                list_lable.append(name_to_lable[name])
    array_to_split = np.array(list_vector)

    return array_to_split, list_lable, name_to_lable


def dataset_create(array_to_split, list_lable):
    """
    function of data division into training and test arrays
    on entry accepts:
        array_to_split -- np.array, array with all data
        list_table ------ list, list of class labels
    returns to output:
        x_trail --------- np.array, an array of data for network training
        y_train --------- np.array, data array for network testing
        x_val ----------- list, list of tags for network training
        y_val ----------- list, list of tags for network testing
    """

    x_train, x_val, y_train, y_val = train_test_split(
        array_to_split, list_lable, test_size=0.3, random_state=42)

    return x_train, y_train, x_val, y_val


def create_model(maxlen, num_category):
    """
    function of creating a model based on the keras Sequential API.
    on entry accepts:
        maxlen -------- int, maximum sequence length
    returns to output:
        model --------- model, compiled model, for training
    """

    model = Sequential()
    model.add(Conv1D(64, 3, activation='elu', input_shape=(maxlen, 200)))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 3, activation='elu'))
    model.add(MaxPooling1D(3))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(GRU(64, return_sequences=True, )))
    model.add(Bidirectional(GRU(64, recurrent_dropout=0.1)))
    model.add(Dense(num_category*3, activation='elu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_category, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Nadam', metrics=['sparse_categorical_accuracy'])

    model.summary()
    return model


def model_fit(x_train, y_train, x_val, y_val, model):
    """
    function for model training.
    on entry accepts:
        x_trail --------- np.array, an array of data for network training
        y_train --------- np.array, data array for network testing
        x_val ----------- list, list of tags for network training
        y_val ----------- list, list of tags for network testing
        model ----------- model, compiled model, for training
    returns to output:
        history --------- , structure with trained model
        and training statistics
    """

    callbacks_list = [EarlyStopping(monitor='val_loss', patience=6),
                      ModelCheckpoint(filepath='GRU_class_checkpoint.h5',
                                      monitor='val_loss', save_best_only=True,)]
    history = model.fit(x_train, y_train,
                        batch_size=5, epochs=50,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks_list)
    return history


def save_model(model, dir_save, name_to_lable, model_name, name_dict):
    """
    save function of the model and the dict matching labels and categories
    on entry accepts:
        model ---------- model, the object of the trained model, to save
        dir_save ------- str, directory to save the model
        name_to_lable -- dict, category label and name mapping dictionary
        model_name ----- str, name of the model to save
    """
    path_model_save = os.path.join(dir_save, "{}.json".format(model_name))
    path_model_weights = os.path.join(dir_save, "{}.h5".format(model_name))
    path_name_lable = os.path.join(dir_save, '{}.pickle'.format(name_dict))

    model_json = model.to_json()
    json_file = open(path_model_save, "w")
    json_file.write(model_json)
    json_file.close()

    model.save_weights(path_model_weights)

    with open(path_name_lable, "wb",) as file_lable:
        pickle.dump(name_to_lable, file_lable)
    print('the model was saved')


def visualise(history):
    """
    function builds a schedule of training models for attribute Accuracy
    on entry accepts:
        history ----------- model learning object
    """
    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'],
             color='r', label='Train')
    plt.show()


def process_of_training(dir_name, model_w2v_name, dir_texts, dir_save,
                        model_name, name_dict):
    """
    function to start the process of creating and training the model
    on entry accepts:
        dir_name --------- str, the directory in which the model will be stored
        model_w2v_name --- str, the name of the model w2v
        dir_txt ---------- str, the directory in which the decomposed
        directory text data for training the network.
    """

    maxlen = 500

    model_w2v = load_w2v_model(dir_name, model_w2v_name)

    array_to_split, list_lable, name_to_lable\
    = create_df_from_dir(dir_texts, maxlen, model_w2v)

    x_train, y_train, x_val, y_val = dataset_create(array_to_split, list_lable)

    num_category = len(name_to_lable)

    model = create_model(maxlen, num_category)

    history_model = model_fit(x_train, y_train, x_val, y_val, model)

    save_model(history_model.model, dir_save, name_to_lable,
               model_name, name_dict)

    visualise(history_model)


def main():
    """
    function to start the learning process
    during execution, data is loaded for training,
    the model is being trained and tested.
    The model is saved to disk.
    """

    dir_name = r'..\models'
    model_w2v_name = 'word2vec_gensim.bin'
    dir_texts = r'..\data\train'
    process_of_training(dir_name, model_w2v_name, dir_texts, dir_name,
                        'conv1D_W2V', 'dict_labels')
#    train model category

    dir_texts = r'..\data\executor'
    process_of_training(dir_name, model_w2v_name, dir_texts, dir_name,
                        'conv1D_W2V_executor', 'dict_labels_executor')
#    train model executor

if __name__ == '__main__':
    main()
