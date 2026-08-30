# House Prices — Regression

Финальный проект по Classic ML: предсказание цены дома (`SalePrice`) на датасете
[House Prices (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).



## Структура репозитория

```
.
├── main.py                     # точка входа: запускает полный пайплайн
├── EDA&experiments.ipynb       # разведочный анализ + эксперименты
├── dnn/                        # DNN-регрессор на PyTorch + sklearn-обёртка
│   ├── DNNLayers.py             # архитектура сети (Linear + BatchNorm + Dropout)
│   ├── DNNSklearn.py            # обёртка под sklearn API (fit/predict)
│   ├── DNNRegressor.py          # альтернативный ручной train-loop с чекпоинтами
│   └── CustomDataset.py         # torch Dataset для табличных данных
└── src/
    ├── config/
    │   ├── global_config.yaml   # общие параметры запуска (путь к данным, таргет, random_state)
    │   ├── param_config.yaml    # гиперпараметры всех моделей
    │   └── omegaconfig.py       # загрузка конфигов через OmegaConf
    ├── data/                    # train.csv, test.csv, sample_submission.csv
    ├── DataUtils/
    │   ├── DataManager.py        # загрузка данных, базовый препроцессинг
    │   ├── base_prepop_funcs.py  # логарифмирование таргета, удаление выбросов, схлопывание категорий
    │   └── leak_safe_preprocess.py # sklearn ColumnTransformer: impute + scale + encoding
    ├── models/
    │   ├── models.py             # реестр всех моделей с гиперпараметрами из конфига
    │   └── ModelsManager.py      # обход моделей (все / только лучшая)
    ├── training/
    │   ├── PipelineRunner.py     # оркестрация: препроцессинг → CV по всем моделям → сохранение лучшей
    │   └── Validater.py          # K-Fold кросс-валидация (общий и отдельно для CatBoost с early stopping)
    └── utils/
        └── utils.py              # сплиты, метрики, сохранение результатов в csv
```

## Конфиг

Все гиперпараметры конфигурируются в `src/config/param_config.yaml`.

## Препроцессинг

- Логарифмирование таргета (`log1p`) из-за скошенного вправо распределения цен.
- Удаление двух выбросов, найденных в EDA (`Id` 524, 1299).
- Дроп малоинформативных колонок (`PoolQC`, `Street`, `Utilities`, `Id`).
- Схлопывание редких категорий и бинаризация сильно несбалансированных категориальных признаков.
- Числовые признаки: импутация средним + `RobustScaler`.
- Категориальные признаки: One-Hot (низкая кардинальность), Ordinal (порядковые), Target Encoding (высокая кардинальность), импутация модой.
- Весь энкодинг обёрнут в `sklearn.ColumnTransformer` и фитится только на train-фолде внутри кросс-валидации — без утечки данных (см. `leak_safe_preprocess.py`).

## Валидация

K-Fold (5 фолдов, см. `get_kf_reg` в `utils.py`), метрика — RMSE на логарифмированном таргете.
Для CatBoost отдельно реализован K-Fold с early stopping (`k_fold_catboost`).

## EDA

Основные выводы разведочного анализа (`EDA&experiments.ipynb`):

- 1460 объектов, 80 признаков, сильно разный масштаб числовых фичей.
- 19 признаков с пропусками, больше всего — `PoolQC`, `Alley`, `Fence`, `MiscFeature`.
- Наибольшая корреляция с таргетом: `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`.
- Сильно скоррелированные пары признаков: `TotRmsAbvGrd`/`GrLivArea`, `GarageCars`/`GarageArea`, `1stFlrSF`/`TotalBsmtSF`, `YearBuilt`/`GarageYrBlt`.
- Таргет имеет длинный правый хвост → логарифмирование.
- Найдено 2 выброса по числовым признакам (id 524, 1299) — удалены из train.

## Как запустить



```bash
pip install -r requirements.txt   # TODO: добавить файл со списком зависимостей
python main.py
```

`main.py` последовательно:
1. загружает и препроцессит `train.csv`;
2. прогоняет K-Fold кросс-валидацию по всем моделям из `models.py`;
3. сохраняет метрики каждой модели в `model_info.csv`;
4. сохраняет лучшую по CV модель в `model.joblib`.

Формирование сабмита (`_create_submission`) и точечная оценка лучшей модели на отложенной выборке
(`_evaluate_best`) реализованы в `PipelineRunner`, но закомментированы в `start_pipline` — раскомментируйте при необходимости.
