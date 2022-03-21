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
 
 ## Пример запуска модели (из корневой папки UnCoRd-VL):
 
 ```
 python main.py --image_dir IMAGE_DIR --questions_file QUESTIONS_DIR/file_with_questions_and_images_indices.json --test_mode True --device "cuda"
 ```
 
 **Замечание**: у меня на сервере возникает ошибка при запуске команды (что-то про дубликаты одной библиотеки), пока не знаю, с чем это связано, но можно исправить следующим образом:
 ```
 export KMP_DUPLICATE_LIB_OK=TRUE
 ```