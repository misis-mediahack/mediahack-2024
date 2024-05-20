# How-To

## Подготовка
1. Установить Python 3.11, CUDA 12
2. Сложить данные (видео и аудиофайлы) куда-нибудь 
3. `python3.11 -m venv .venv`
4. `source .venv/bin/activate`
5. `pip install -r requirements.txt`

## Генерация фичей
1. Убедитесь, что вы выполнили блок Подготовка
2. Для того чтобы сделать предсказания или обучить модель, нужно сгенерировать фичи
3. Генерация фичей транскрибаций:
```bash
python -m mediahack.s1_extract_audio --video-dir ./data --transcription-path ./data/transcriptions.csv
```
Аргумент --video-dir отвечает за папку с видео и аудио, а --transcription-path отвечает за путь к файлу, куда будут записываться транскрибации 
4. Генерация фичей из CLIP:
```bash
python -m mediahack.s1_extract_clip --video-dir ./data --embed-dir ./data/clip_embeddings
```
Аргумент --video-dir отвечает за папку с видео и аудио, а --embed-dir отвечает за путь к папке, куда будут записываться фичи CLIP.
5. Генерация фичей OCR:
```bash
python -m mediahack.s1_extract_ocr --video-dir ./data --ocr-dir ./data/ocr_data
```
Аргумент --video-dir отвечает за папку с видео и аудио, а --ocr-dir отвечает за путь к папке, куда будут записываться фичи OCR


## Обучение модели (можно пропустить, если файл с моделью уже есть)
Убедитесь, что вы выполнили блок Подготовка, а также сгенерировали фичи для тренировочных данных (см. блок Генерация фичей)

Обучить модель можно, запустив простую команду
```bash
python -m mediahack.s2_train_model --transcription-path ./data/transcriptions.csv --clip-dir ./data/clip_embeddings --ocr-dir ./data/ocr_data --target-path ./data/train_segments.csv
```
Аргумент --transcription-path отвечает за путь к файлу с фичами транскрибаций, --clip-dir отвечает за путь к папке с фичами CLIP, --ocr-dir отвечает за путь к папке с фичами OCR, --target-path отвечает за путь к таргетам

После обучения метрики можно посмотреть в tensorboard: `tensorboard --logdir runs`

После обучения файл с моделью будет сохранен в: `./checkpoint/train/save-*/model.safetensors`


## Предсказание
Предобученную модель можно сказать [здесь](https://drive.google.com/file/d/1IlaIjmAg5pV6RWhnIg6yce11VyrUTU4D/view?usp=sharing)

Убедитесь, что вы выполнили блок Подготовка, а также сгенерировали фичи для тестовых данных (см. блок Генерация фичей)

Убедитесь, что вы обучили модель или имеете файл с обученной моделью

Выполнить предсказание на тестовых данных можно, запустив простую комамнду
```bash
python -m mediahack.s3_infer --video-dir ./data-test --model-file ./model.safetensors --transcription-path ./data-test/transcriptions.csv --clip-dir ./data-test/clip_embeddings --ocr-dir ./data-test/ocr_data --out-path ./predict.csv
```
Аргумент --video-dir отвечает за путь к директории с промо, --model-file отвечает за путь к файлу с моделью, --transcription-path отвечает за путь к файлу с фичами транскрибаций, --clip-dir отвечает за путь к папке с фичами CLIP, --ocr-dir отвечает за путь к папке с фичами OCR, --out-path отвечает за путь к файлу, куда будут сохранены предсказация

Также можно задать аргументы --batch-size (число) и --device (cuda или cpu), чтобы изменить количество одновременно обрабатываемых данных (если они не влазят в память), либо изменить устройство (если нет GPU)

После окончания предсказания, будет создан файл, имеющий две колонки - идентификатор промо и предсказанный класс. Путь указывается аргументом --out-path


## Дашборд
 
Запустить дашборд можно командой
```bash
 python -m mediahack.dashboard --dashboard-data data/dashboard_data.csv --segment-dict data/segment_dict.xlsx --segment-predictions data/train_segments.csv
```
Аргументы --dashboard-data, --segment-dict, --segment-predictions отвечают за пути к данным дашборда, названиям классов и предсказаниям классов соответственно

Также можно указать аргумент --port (номер порта), чтобы изменить порт по умолчанию.

По умолчанию дашборд после запуска будет доступен по адресу `localhost:6008`
