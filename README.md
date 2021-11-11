# Пример использования Metaflow

Реализуем [Mobilevit](https://keras.io/examples/vision/mobilevit/) с помощью Metaflow


### Файлы
- `src/model.py` - содержит функцию, генерирующую MobileViT модель
- `scr/preprocessing` - функции, обрабатывающие изображения подаваемые на вход модели
- `src/main.py` - файл, содержащий MetaFlow Pipeline, состоящий из следующих шагов:
    - **start** - установка гиперпараметров
    - **load_datasets** - загрузка датасета и сохранение в виде Pandas DataFrame
    - **preprocessing** - обработка изображений и сохранение данных после обработки в виде Pandas DataFrame
    - **train_and_save_model** - обучение модели и ее сохранение
    - **end** - окончание пайплайна

### Инструкция по запуску

```Bash
python3 src/main.py --environment=conda run
```