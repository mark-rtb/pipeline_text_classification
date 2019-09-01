"""
augmentation module for texts by translating them into another language and Vice versa
"""
import os
from googletrans import Translator


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


def save_txt(text_finaly, dir_name, file_name):
    """ Функция сохранения текстовых файлов.
    На вход принимает строку.
    Создает файл .txt со случайным именем в рабочей дирректории """

    file = open("{}.txt".format(os.path.join(dir_name, file_name)), "w", encoding='utf-8')
    file.write(text_finaly)
    file.close()


def augment_text(text_to_tans, list_lang):
    """
    function translates text into several foreign languages and then back
    on entry receives:
        text_to_tans ------ str, the text to augment
        list_lang --------- list, list of languages to translate
    returns to output:
        list_text_lang ---- list, a list of the texts translated into other
        languages and back
    """
    translator = Translator()
    list_text_lang = []
    for lang in list_lang:
        tr_str = translator.translate(text_to_tans, src='ru', dest=lang)
        rus_tr_str = translator.translate(tr_str.text, src=lang, dest='ru')
        list_text_lang.append(rus_tr_str.text)
    return list_text_lang


def translate_texts(dir_train, list_lang):
    """
    function reads the file from disk, augmentation and preservation
    augmented data
    on entry accepts:
        air_train -------- str, directory which contains the files
        list_lang -------- list, list of languages to translate
    """
    
    list_train_dir = os.listdir(dir_train)
    for name in list_train_dir:
        print('аугментация: {}'.format(name))
        dir_name = os.path.join(dir_train, name)
        list_files = os.listdir(dir_name)
        if len(list_files)<20:
            for file in list_files:
                if file[-4:] == '.txt':
                    text_to_tans = load_texts_for_dataset(file, dir_name)
                    list_text_save = augment_text(text_to_tans, list_lang)
                    for num_lang, text_save in enumerate(list_text_save):
                        save_txt(text_save, dir_name, str(num_lang)+file)


def main():
    """
    this function starts the data augmentation process
    """
    
    list_lang = ['en', 'hy', 'ko', 'zh-cn', 'nl', 'eo', 'fr', 'de', 'hi', 'ja']
    dir_train = r'..\data\train'
    print('аугментация текстов')
    translate_texts(dir_train, list_lang)
    
    
if __name__ == '__main__':
    main()
