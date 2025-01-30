# SHIFT train project
Этот проект - бейзлайн для обучения модели классификации, ваша задача - улучшить его.

## Окружение
Для начала склонируйте проект и настройте окружение
```bash
git clone https://github.com/Dragon181/SHIFT-intensive.git
cd SHIFT-intensive
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Запуск проекта
Перед началом разбейте датасет на train, val, test для лучшего обучения.
Не забудьте поправить пути до новых датасетов в [конфигурации](conf/data/sign_train.yaml)
```bash
python3 train.py
```
В репозитории настроено логирование результатов обучения. Вы можете смотреть, как изменялись ваши метрики качества на протяжении обучения.
Для этого запустите в командной строке:

```bash
tensorboard --logdir=tensorboard
```
После этого перейдите по адресу http://localhost:6006/, там вы увидите, как изменяются метрики (accuracy, f1, precision, recall)

## С чего начать?
Для начала просмотрите параметры конфигураций [configs](conf/).
Так же внимательно изучите, как формируется [dataloader](srcs/data_loader/data_loaders.py)
По умолчанию используется модель mobilenet_v3_small, попробуйте что-то лучше

## Полезные ссылки
- Туториалы по обучению на Pytorch - https://pytorch.org/tutorials/
- Документация hydra - https://hydra.cc/docs/intro/