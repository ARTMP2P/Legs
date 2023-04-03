# *1. Запуск виртуального окружения*
```python
    source venv/bin/activate
```
# *3. Установка библиотек*
```python
    pip install -r req.txt
```
# *2. Запуск файла*
```python
    python3 main.py
```
Данный код используется для обучения генеративно-состязательной сети (GAN) на заданном наборе данных. Он импортирует необходимые модули и функции из других файлов, определяет необходимые параметры, в том числе количество эпох обучения и размер изображения, а также создает и компилирует модели дискриминатора, генератора и GAN. Затем он запускает функцию обучения с заданными параметрами и набором данных.

Также код включает настройку работы TensorFlow с GPU для эффективного использования ресурсов вычислительной машины.

# 4. Дополнительные функции:

## 1. Парсер датасета.
Парсер предназначен для изменения размеров изображений в формате PNG, которые находятся в указанной входной директории и ее подпапках, до размеров 1024х1024 пикселей с сохранением пропорций и обрезкой до квадрата. Измененные изображения сохраняются в новых директориях, создаваемых в указанной выходной директории в соответствии с исходными директориями, содержащими файлы изображений.

Программа запускается через командную строку с помощью Python и принимает два обязательных аргумента: путь к входной директории с изображениями и путь к выходной директории для сохранения измененных изображений.

Для обработки изображений программа создает пул потоков, количество потоков равно количеству доступных процессоров, затем проходит по всем файлам PNG во входной директории и ее подпапках, добавляя задачи в пул потоков. Каждая задача заключается в изменении размера изображения и сохранении его в новой папке, соответствующей структуре исходной папки. В конце работы все потоки завершаются.

Функция process_file принимает два аргумента: путь к исходному файлу изображения и путь к директории, в которой необходимо сохранить измененное изображение. Функция resize_png изменяет размер изображения и сохраняет его в указанной директории.

Запуск:
```python
    python3 parser.py
```


# 5. Результат выполнения по каждой эпохе
```
3145left
3.5756945610046387 |(суммарный mIoU по всем ракурсам модели стопы)
3091right
2.9930591583251953
3200right
3.2172441482543945
287left
3.704690933227539
3122right
3.207266330718994
3057left
2.8534770011901855
```
