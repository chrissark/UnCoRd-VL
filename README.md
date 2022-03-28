# UnCoRd-VL

## Настройка среды
Выполните команды, перечисленные ниже.
```
git clone https://github.com/chrissark/UnCoRd-VL
```
1. Создание среды anaconda:

   ```
   conda create env --name uncord-vl
   conda activate uncord-vl 
   conda install pytorch=1.10.2 torchvision=0.11.* cudatoolkit -c pytorch
   ```
2. Установка зависимостей для VL-BERT. Внутри папки [vlbert](./vlbert) находятся файлы из репозитория с оригинальной моделью [VL-BERT](https://github.com/jackroos/VL-BERT).

   ```
   cd vlbert
   pip install -r requirements.txt
   pip install Cython
   pip install pyyaml==5.4.1
   ```
3. Сборка библиотеки для VL-BERT:
   
   ```
   ./scripts/init.sh
   ```
 
 ## Подготовка данных
 Для запуска UnCoRd-VL на собственном датасете нужно подготовить следующие данные:
 1. Checkpoint модели **Question-to-graph** - поместить в папку [ende_ctranslate2](./ende_ctranslate2).
 2. Checkpoint модели **Faster-RCNN** - поместить в папку [estimators](./estimators).
 3. Checkpoint **VL-BERT** - поместить в папку [vlbert/model/pretrained_model](./vlbert/model/pretrained_model).
 4. Список названий свойств и их возможных значений
 
    Нужно подготовить файл, в котором перечислены все возможные категориальные свойства в датасете и значения, которые они могут принимать. Каждую строку нужно заполнить в следующем формате:
    
    ```
    название_свойства значение_1 ... значение_n
    ```
    
    [Пример файла свойств для CLEVR.](properties_file.txt)
 5. Словарь с ответами для VL-BERT 
 
    Ответы для VL-BERT -- это множество, состоящее из всех возможных значений всех свойств из заданного датасета, а также из слов 'yes' и 'no'. [Пример словаря ответов для CLEVR.](answer_vocab_file.txt)
 
 6. Директория с изображениями
 7. JSON-файл с индексами вопросов, текстами вопросов, а также с индексами изображений, которые соответствуют этим вопросам (ответы необязательны для тестового режима).

    ```
    {'questions': [{'question_index': 0, 'question': 'What is the color of the large metal sphere?', 'image_index': 0, 'answer': 'brown'}, ... ]}
    ```
    Функции для извлечения вопросов в файле [dataset.py](dataset.py) (можно менять под свои данные).
    
 Кроме того, для работы с VL-BERT нужно скачать предобученные [BERT](https://drive.google.com/file/d/14VceZht89V5i54-_xWiw58Rosa5NDL2H/view?usp=sharing) и [ResNet-101](https://drive.google.com/file/d/1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE/view?usp=sharing) и поместить их в папки  [vlbert/model/pretrained_model/bert-base-uncased](./vlbert/model/pretrained_model/bert-base-uncased) и [vlbert/model/pretrained_model](./vlbert/model/pretrained_model) соответственно.
 
 ## Пример запуска модели (из корневой папки UnCoRd-VL):
 
 ```
 python main.py --image_dir IMAGE_DIR \
 --questions_file QUESTIONS_DIR/file_with_questions_and_images_indices.json \
 --test_mode True --device "cuda" \
 --answer_vocab_file answer_vocab_file.txt \
 --properties_file properties_file.txt
 ```
 
 **Замечание**: у меня на сервере возникает ошибка при запуске команды (что-то про дубликаты одной библиотеки), пока не знаю, с чем это связано, но можно исправить следующим образом:
 ```
 export KMP_DUPLICATE_LIB_OK=TRUE
 ```