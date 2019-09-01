**Общее описание логики работы решения**

Решение основанно на применении глубоких одномерных сверточных и двунаправленных
рекуррентных нейронны сетей.

Для упрощения процесса внедрения был разработан pipline обучения всех элементов 
решения. Для его запуска нужно выполнить файл: category_classification_train.py 
В процессе работы скрипта, создаются все необходимые служебные файлы, обучаются 
модели классификации и лучшие результаты сохраняются на диск. 
Чтобы запустить предсказание классов и вывод информации о содержании текста нужно
выполнить файл: run_prediction.py
В процессе выполнения скрипт загружает модели и необходимые словари, выполняет
предсказание. 

Модель использует word2vec погружение слов. Модель word2vec обучена на  
датасете состоящем из обращений граждан, твитов, статей википедии в пропорции:
50%|20%|30%. Если в процессе развертывания понадобится дообучить представления 
словб можно запустить скрипт: word_2_vec_auto_class.py 

Так как данных для обучения катастрофически мало, тренировочный набор был дополнен
аугментированными текстами, аугментация выполняется путем перевода текста на 
иностранный язык и затем обратно. Если понадобится дополнительная аугментация 
можно запустить скрипт: google_transl_api.py, который выполняет перевод файлов,
если в категории их меньше 20. 

*Для удобства развертывания создан файл setup.py,* он содержит все необходимые 
для запуска зависимости. Чтобы запустить файл, необходимо перейти в дирректорию
с файлом и выполнить команду python setup.py install


**Требования к окружению для запуска продукта**
Платформа: Скрипт тестировался на платформе windows
Используемый язык python 3.6

**Сценарий сборки и запуска проекта**
Скопируйте репозиторий на локальную машину,
перейдите в дирректорию с файлом setup.py
выполните команду:

`python setup.py install`

после того, как установятся все зависимости, перейдите в дирректорию \app
и выполните файл: word_2_vec_auto_class.py

`python run_prediction.py`

Скрипт загрузит предобученные модели и выведет в консоль предсказания.
Если требуется переобучить модели (если появятся новые данные) запустите файл: 
category_classification_train.py

`python category_classification_train.py`

Будет выполнено обучение моделей и выведены графики правильности на тестовых данных


