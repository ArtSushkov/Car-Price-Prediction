# === Импорт библиотек ===

# Стандартная библиотека Python
import warnings
from itertools import combinations

# Сторонние библиотеки для анализа данных и визуализации
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative
import seaborn as sns

# Сторонние библиотеки для статистического анализа
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Сторонние библиотеки для машинного обучения и препроцессинга
from category_encoders import CatBoostEncoder
from phik.report import plot_correlation_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# Подавление предупреждений (для чистоты вывода в ноутбуке)
warnings.filterwarnings('ignore')


# === Задание констант и списков признаков ===

# константа RANDOM_STATE
RANDOM_STATE = 42

# константа TEST_SIZE
TEST_SIZE = 0.25

# Списки признаков для разных типов обработки
binary_features = ['Gearbox', 'Repaired']  # Бинарные признаки (да/нет)
ohe_features = ['VehicleType', 'FuelType']  # Признаки для One-Hot Encoding
catboost_features = ['brand_model', 'PostalRegion']  # Признаки для CatBoostEncoder
numeric_features = ['Kilometer', 'Power', 'RegistrationYear']  # Числовые признаки


# === Собственные функции и класс ===

class FeatureUnionWithCatBoost(BaseEstimator, TransformerMixin):
    """Объединяет предобработку основных признаков и CatBoost-кодирование."""
    def __init__(self, preprocessor=None, catboost_pipe=None, catboost_features=None):
        self.preprocessor = preprocessor
        self.catboost_pipe = catboost_pipe
        self.catboost_features = catboost_features
        self.ohe_features_count_ = 0
        self.numeric_features_count_ = 0
        self.catboost_features_count_ = 0

    def fit(self, X, y=None):
        if self.preprocessor is not None:
            self.preprocessor.fit(X, y)
            # Получаем количество фичей после OHE
            ohe_transformer = self.preprocessor.named_transformers_['ohe']
            self.ohe_features_count_ = ohe_transformer.named_steps['ohe'].get_feature_names_out().shape[0]
            self.numeric_features_count_ = len(numeric_features)

        if self.catboost_pipe is not None and self.catboost_features is not None:
            self.catboost_pipe.fit(X[self.catboost_features], y)
            self.catboost_features_count_ = len(self.catboost_features)
        return self

    def transform(self, X):
        X_processed = self.preprocessor.transform(X) if self.preprocessor is not None else np.zeros((X.shape[0], 0))
        if self.catboost_pipe is not None and self.catboost_features is not None:
            X_catboost = self.catboost_pipe.transform(X[self.catboost_features])
            return np.hstack([X_processed, X_catboost])
        return X_processed

    def get_feature_counts(self):
        return {
            'OneHot features': self.ohe_features_count_,
            'Numeric features': self.numeric_features_count_,
            'CatBoost features': self.catboost_features_count_,
            'Total features': self.ohe_features_count_ + self.numeric_features_count_ + self.catboost_features_count_
        }

    def set_params(self, **params):
        if self.preprocessor is not None:
            preprocessor_params = {k[14:]: v for k, v in params.items() if k.startswith('preprocessor__')}
            self.preprocessor.set_params(**preprocessor_params)

        if self.catboost_pipe is not None:
            catboost_params = {k[15:]: v for k, v in params.items() if k.startswith('catboost_pipe__')}
            self.catboost_pipe.set_params(**catboost_params)

        return self


def main(X_train, y_train, scoring, model, param_grid, n_splits=3, n_iter=10, weights_train=None):
    """
    Основная функция для настройки и обучения модели с обработкой различных типов признаков
    и поиском лучших гиперпараметров с помощью RandomizedSearchCV.

    Параметры:
    ----------
    X_train : pandas.DataFrame
        Обучающие данные (признаки).
    y_train : pandas.Series или numpy.ndarray
        Целевая переменная.
    scoring : dict или str
        Метрики для оценки модели.
    model : sklearn estimator
        Модель машинного обучения для обучения.
    param_grid : dict
        Сетка параметров для поиска.
    n_splits : int
        Количество фолдов в StratifiedKFold (по умолчанию 3).
    n_iter : int
        Количество итераций поиска RandomizedSearchCV (по умолчанию 10).     
    weights_train : pandas.Series или numpy.ndarray, optional
        Веса образцов для обучения (по умолчанию None).

    Возвращает:
    -----------
    tuple (RandomizedSearchCV, float)
        Обученный объект RandomizedSearchCV и среднее время обучения одной модели в секундах.

    Пример использования:
    ---------------------
    >>> # Импорт необходимых библиотек
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.metrics import make_scorer, mean_squared_error
    >>> import numpy as np
    >>>
    >>> # Определение типов признаков
    >>> binary_features = ['binary_feat1', 'binary_feat2']
    >>> ohe_features = ['category_feat1', 'category_feat2']
    >>> numeric_features = ['numeric_feat1', 'numeric_feat2']
    >>> catboost_features = ['catboost_feat1', 'catboost_feat2']
    >>>
    >>> # Создание тестовых данных
    >>> data = {
    ...     'binary_feat1': [0, 1, 0, 1],
    ...     'binary_feat2': [1, 0, 1, 0],
    ...     'category_feat1': ['A', 'B', 'A', 'C'],
    ...     'category_feat2': ['X', 'Y', 'Z', 'X'],
    ...     'numeric_feat1': [1.2, 3.4, 5.6, 7.8],
    ...     'numeric_feat2': [0.5, 1.5, 2.5, 3.5],
    ...     'catboost_feat1': ['high', 'low', 'medium', 'high'],
    ...     'catboost_feat2': ['red', 'blue', 'green', 'red']
    ... }
    >>> X_train = pd.DataFrame(data)
    >>> y_train = pd.Series([10, 20, 15, 25])
    >>>
    >>> # Настройка модели и параметров
    >>> model = RandomForestRegressor(random_state=42)
    >>> param_grid = {
    ...     'regressor__n_estimators': [50, 100, 200],
    ...     'regressor__max_depth': [None, 5, 10]
    ... }
    >>> scoring = {
    ...     'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
    ...                        greater_is_better=False),
    ...     'mae': 'neg_mean_absolute_error'
    ... }
    >>>
    >>> # Вызов функции
    >>> search_cv, train_time = main(
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     scoring=scoring,
    ...     model=model,
    ...     param_grid=param_grid
    ... )
    >>>
    >>> # Использование лучшей модели
    >>> best_model = search_cv.best_estimator_
    >>> predictions = best_model.predict(X_train)
    """

    # Пайплайн для обработки бинарных и категориальных признаков (One-Hot Encoding)
    ohe_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False)),
        ('imputer_after', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Пайплайн для обработки числовых признаков
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Главный предобработчик данных
    preprocessor = ColumnTransformer([
        ('ohe', ohe_pipe, binary_features + ohe_features),
        ('numeric', numeric_pipe, numeric_features)
    ], remainder='drop')

    # Пайплайн для CatBoostEncoder
    catboost_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('catboost', CatBoostEncoder())
    ])

    # Создание объединенного обработчика признаков
    feature_union = FeatureUnionWithCatBoost(
        preprocessor=preprocessor,
        catboost_pipe=catboost_pipe,
        catboost_features=catboost_features
    )

    # Финальный пайплайн
    pipe_final = Pipeline([
        ('feature_union', feature_union),
        ('regressor', model)
    ])

    # Параметры для поиска
    param_grid = param_grid

    # Используем стратификацию, так как цена имеет неравномерное распределение
    # Создаем бины для стратификации (10 квантилей)
    y_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')

    # Настройка стратифицированной кросс-валидации
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Настройка RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        cv=inner_cv.split(X_train, y_bins),  # Передаем разбиение с бинами
        scoring=scoring,
        refit='rmse',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_iter=n_iter,
        error_score='raise',
        return_train_score=True
    )

    # Обучение модели
    randomized_search.fit(X_train, y_train, regressor__sample_weight=weights_train)

    # Извлекаем среднее время обучения одной модели (без подбора гиперпараметров и предсказаний)
    # Берем среднее время по всем фолдам и всем итерациям
    mean_fit_time = np.mean(randomized_search.cv_results_['mean_fit_time'])
    
    print(f"Mean training time per model: {mean_fit_time:.2f} seconds")
    print("Best parameters:", randomized_search.best_params_)
    print("Best RMSE (CV):", abs(randomized_search.best_score_))

    return randomized_search, mean_fit_time


def plot_sunburst_distribution(df, group_features, target_feature, show_unique_values=True):
    """
    Строит Sunburst-диаграмму распределения одного признака
    внутри комбинаций других признаков.

    Args:
        - df: pandas.DataFrame — исходный датафрейм
        - group_features: list строк — признаки для группировки (может быть 1 и более)
        - target_feature: str — признак, распределение которого исследуем
        - show_unique_values: bool (по умолчанию True) — выводить ли уникальные значения признаков

    Examples:
        >>> plot_sunburst_distribution(df, ['VehicleType'], 'FuelType')
        >>> plot_sunburst_distribution(df, ['Brand', 'Model'], 'FuelType')
        >>> plot_sunburst_distribution(df, ['Brand', 'Model', 'VehicleType'], 'FuelType')
    """

    # Группировка данных
    grouped = df.groupby(group_features)[target_feature].value_counts().reset_index(name='count')

    # Вывод уникальных значений (если включён)
    if show_unique_values:
        print("Уникальные значения в признаках:")
        for feature in group_features + [target_feature]:
            print(f"{feature}: {grouped[feature].unique()}")

    # Переименование столбцов
    grouped.columns = group_features + [target_feature, 'count']

    # Получаем все уникальные значения целевого признака для цветовой палитры
    unique_values = grouped[target_feature].unique()

    # Берём несколько стандартных палитр
    Bold = getattr(qualitative, 'Bold', [])
    Dark24 = getattr(qualitative, 'Dark24', [])
    Light24 = getattr(qualitative, 'Light24', [])
    Prism = getattr(qualitative, 'Prism', [])
    Vivid = getattr(qualitative, 'Vivid', [])
    Pastel1 = getattr(qualitative, 'Pastel1', [])
    Set3 = getattr(qualitative, 'Set3', [])

    # Собираем расширенную палитру
    extended_palette = Bold + Dark24 + Light24 + Prism + Vivid + Pastel1 + Set3

    # Если недостаточно цветов — повторяем
    color_palette = (extended_palette * (len(unique_values) // len(extended_palette) + 1))[:len(unique_values)]

    # Построение Sunburst диаграммы
    fig = px.sunburst(
        grouped,
        path=group_features + [target_feature],
        values='count',
        title=f"Распределение признака '{target_feature}' по признакам {', '.join(group_features)}",
        width=1200,
        height=1000,
        color=target_feature,  # Цвет зависит от значения target_feature
        color_discrete_sequence=color_palette  # Используем расширенную палитру
    )

    # Настройка отображения текста
    fig.update_traces(
        textinfo='label+percent parent',  # Название + доля относительно родителя
        insidetextorientation='radial'     # Радиальное расположение текста
    )

    # Отображение графика
    fig.show()


def plot_distribution_with_boxplot(df, features, target_col, category_order=None, bins=None, auto_bins=False):
    """Визуализирует распределения признаков с разделением по категориям целевой переменной.

    Автоматически преобразует числовые целевые переменные в категориальные,
    если они содержат ≤10 уникальных значений. Это преобразование не влияет на исходный датафрейм.

    Создает сетку графиков:
    - Для каждого признака отображаются 2 строки:
      1) Гистограмма с наложениями категорий (stacked) и KDE-кривой
      2) Горизонтальный boxplot, где категории размещены по оси y
    - Цвета категорий согласованы между графиками и указаны в легенде
    - Графики автоматически размещаются в сетке (до 4 столбцов)
    - Убираются пустые оси для незаполненных ячеек сетки
    - Настройки: сетка, поворот меток, оптимизация макета

    Args:
        df (pd.DataFrame): DataFrame с данными
        features (List[Tuple[str, str]]): Список кортежей (колонка, человекочитаемая метка)
        target_col (str): Название целевой колонки. Если тип числовой и содержит ≤10 уникальных значений,
                          будет преобразована в категориальную внутри функции
        category_order (Optional[List[str]]): Список категорий в нужном порядке. Если None,
            используется отсортированный порядок (по умолчанию None)
        bins (Optional[int]): Количество корзин для гистограмм. Если None и auto_bins=False,
            используется значение по умолчанию в sns.histplot (по умолчанию None)
        auto_bins (bool): Если True, для каждого признака количество корзин будет определяться
            автоматически с помощью правила Фридмана-Дьякониса (по умолчанию False)

    Notes:
        - features ожидает кортежи (колонка, человекочитаемая метка)

    Examples:
        >>> # Пример 1: Базовое использование с автоматическим определением категорий
        >>> features = [('age', 'Возраст'), ('income', 'Доход')]
        >>> plot_distribution_with_boxplot(df, features, 'gender')

        >>> # Пример 2: Фиксированное количество корзин для всех гистограмм
        >>> plot_distribution_with_boxplot(df, features, 'gender', bins=20)

        >>> # Пример 3: Автоматический подбор корзин для каждого признака
        >>> plot_distribution_with_boxplot(df, features, 'gender', auto_bins=True)

        >>> # Пример 4: Указание порядка категорий для целевой переменной
        >>> category_order = ['low', 'medium', 'high']
        >>> plot_distribution_with_boxplot(df, features, 'risk_level', category_order=category_order)

        >>> # Пример 5: Комбинирование параметров
        >>> features = [('age', 'Возраст'), ('income', 'Доход'), ('score', 'Кредитный балл')]
        >>> plot_distribution_with_boxplot(
        ...     df,
        ...     features,
        ...     'loan_status',
        ...     category_order=['approved', 'rejected', 'pending'],
        ...     auto_bins=True
        ... )
    """
    # Проверка и преобразование целевой переменной
    target_series = df[target_col]
    unique_count = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    # Создаем временную целевую переменную
    if is_numeric and unique_count <= 10:
        # Преобразуем числовые значения в строки для категоризации
        transformed_target = target_series.astype(str)
        print(f"Целевой признак '{target_col}' преобразован в категориальный. "
              f"Уникальные значения: {sorted(transformed_target.unique())}")
    else:
        transformed_target = target_series.copy()

    # Получаем категории в нужном порядке (удалив строки с пропусками, если они имеются)
    if category_order is None:
        if is_numeric and unique_count <= 10:
            # Сортируем оригинальные числовые значения, затем преобразуем в строки
            numeric_categories = sorted(df[target_col].dropna().unique())
            categories = [str(val) for val in numeric_categories]
        else:
            # Для нечисловых или числовых с >10 уникальными значениями
            categories = sorted(transformed_target.dropna().unique())
    else:
        categories = category_order

    # Подготовка цветовой палитры и элементов легенды
    palette = plt.cm.Paired(np.linspace(0, 1, len(categories)))
    category_to_color = {cat: color for cat, color in zip(categories, palette)}

    legend_elements = [
        Patch(facecolor=color, label=str(cat), alpha=0.6)
        for cat, color in category_to_color.items()
    ]

    # Настройка сетки графиков
    n_features = len(features)
    ncols = min(4, n_features)  # количество столбцов в сетке графиков (не более 4)
    rows_per_feature = 2  # количество строк на каждый признак (2 строки: гистограмма и boxplot)
    nrows = (n_features + ncols - 1) // ncols * rows_per_feature  # общее количество строк

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 3 * nrows),
        squeeze=False  # гарантирует двумерный массив axes
    )

    # Функция для вычисления оптимального числа корзин по правилу Фридмана-Дьякониса
    def calculate_fd_bins(data):
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1/3))
        if bin_width == 0:
            return 30  # значение по умолчанию, если все данные одинаковы
        return int(np.ceil((data.max() - data.min()) / bin_width))

    # Цикл по признакам для создания графиков
    for i, (feature_col, feature_label) in enumerate(features):
        col = i % ncols  # столбец для текущего признака
        row_base = (i // ncols) * rows_per_feature  # базовая строка для текущего признака

        # Определение количества корбин для текущего признака
        current_bins = bins
        if auto_bins and pd.api.types.is_numeric_dtype(df[feature_col]):
            current_bins = calculate_fd_bins(df[feature_col].dropna())
            print(f"Для признака '{feature_col}' автоматически выбрано {current_bins} корзин")

        # Гистограмма
        ax_hist = axes[row_base, col]
        sns.histplot(
            data=df,
            x=feature_col,
            hue=transformed_target,  # Используем преобразованный целевой признак
            kde=True,
            multiple='stack',
            palette=category_to_color,
            ax=ax_hist,
            legend=False,
            bins=current_bins
        )

        # Настройка гистограммы
        ax_hist.set_title(f'Распределение {feature_label}', fontsize=10)
        ax_hist.set_xlabel('')

        if i == 0:
            ax_hist.set_ylabel('Частота', fontsize=9)
        else:
            ax_hist.set_ylabel('')

        if i == 0:
            ax_hist.legend(
                handles=legend_elements,
                title=target_col,
                fontsize=8,
                title_fontsize=8,
                loc='upper right'
            )

        ax_hist.grid(axis='y', alpha=0.3)
        ax_hist.tick_params(axis='x', labelrotation=0)

        # Boxplot
        ax_box = axes[row_base + 1, col]
        sns.boxplot(
            data=df,
            x=feature_col,
            y=transformed_target,  # Используем преобразованный целевой признак
            order=categories,
            palette=category_to_color,
            orient='h',
            ax=ax_box,
            width=0.6
        )

        # Настройка boxplot
        if i == 0:
            ax_box.tick_params(axis='y', labelrotation=45, labelsize=8)
        else:
            ax_box.set_yticklabels([])

        ax_box.set_xlabel(feature_label, fontsize=9)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.tick_params(axis='x', labelrotation=0)
        ax_box.set_ylabel('')

    # Скрытие пустых осей
    for row in range(nrows):
        for col in range(ncols):
            current_idx = (row // rows_per_feature) * ncols + col
            if current_idx >= n_features:
                axes[row, col].set_visible(False)

    # Настройка макета и отображение графиков
    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    plt.show()


def plot_features_by_class(df, target, features=None, facet_by=None, plot_type='scatter',
                         kde_samples=1000, kde_levels=5,
                         scatter_alpha=0.2, scatter_size=5, kde_alpha=0.5,
                         legend_location='best', legend_title=None,
                         figsize_multiplier=1.0, warn_singular=False,
                         facet_col_wrap=3):
    """
    Визуализация распределения числовых признаков по классам целевой переменной с возможностью фасетирования.
    Поддерживает два типа графиков: scatter и KDE для любого количества классов.

    Args:
    ----------
        df : pd.DataFrame
            Входной DataFrame с данными
        target : str
            Название целевой переменной
        features : list, optional
            Список признаков для анализа (по умолчанию все числовые)
        facet_by : str, optional
            Название дополнительного признака для фасетирования (категориальный или дискретный числовой)
        plot_type : {'scatter', 'kde'}, default='scatter'
            Тип визуализации:
            - 'scatter' - точечный график
            - 'kde' - ядерная оценка плотности
        kde_samples : int, default=1000
            Количество точек для подвыборки в KDE
        kde_levels : int, default=5
            Количество уровней изолиний в KDE
        scatter_alpha : float, default=0.2
            Прозрачность точек (0-1)
        scatter_size : float, default=5
            Размер точек
        kde_alpha : float, default=0.5
            Прозрачность KDE (0-1)
        legend_location : str, default='best'
            Позиция легенды
        legend_title : str, optional
            Заголовок легенды
        figsize_multiplier : float, default=1.0
            Множитель размера фигуры
        warn_singular : bool, default=False
            Показывать предупреждения для сингулярных матриц
        facet_col_wrap : int, default=3
            Количество колонок в фасетной сетке

    Examples:
    ---------------------
        Пример 1: Scatter plot с фасетированием по дополнительному признаку
        >>> plot_features_by_class(df, 'target_column', facet_by='region')

        Пример 2: KDE plot с фасетированием и настройками визуализации
        >>> plot_features_by_class(df, 'target', facet_by='gender',
        ...                      plot_type='kde', kde_alpha=0.7,
        ...                      legend_title='Customer Segments')
    """

    # Проверка параметров
    if plot_type not in ['scatter', 'kde']:
        raise ValueError("plot_type должен быть 'scatter' или 'kde'")

    if not 0 <= scatter_alpha <= 1 or not 0 <= kde_alpha <= 1:
        raise ValueError("Параметры alpha должны быть между 0 и 1")

    # Выбор числовых признаков
    if features is None:
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        numeric_features = [col for col in numeric_features if df[col].nunique() > 1]
        if target in numeric_features:
            numeric_features.remove(target)
    else:
        numeric_features = [col for col in features
                          if col in df.columns
                          and pd.api.types.is_numeric_dtype(df[col])
                          and df[col].nunique() > 1]
        if target in numeric_features:
            numeric_features.remove(target)

    if len(numeric_features) < 2:
        raise ValueError("Необходимо минимум 2 числовых признака с вариацией")

    # Проверка параметра facet_by
    if facet_by is not None:
        if facet_by not in df.columns:
            raise ValueError(f"Признак {facet_by} не найден в DataFrame")

        # Преобразуем числовой признак в категориальный, если уникальных значений немного
        if pd.api.types.is_numeric_dtype(df[facet_by]) and df[facet_by].nunique() > 10:
            print(f"Предупреждение: Признак {facet_by} имеет много уникальных значений ({df[facet_by].nunique()}). "
                  "Рекомендуется использовать категориальный признак или дискретный числовой с небольшим количеством уникальных значений.")

    # Получаем уникальные классы целевой переменной
    classes = df[target].unique()
    n_classes = len(classes)

    # Генерация пар признаков
    pairs = list(combinations(numeric_features, 2))
    n_plots = len(pairs)

    # Цвета для классов
    colors = sns.color_palette("husl", n_colors=n_classes)
    legend_title = legend_title or target

    # Функция для построения одного графика
    def plot_single(ax, x_col, y_col, data):
        handles = []
        labels = []
        empty_plot = True

        for i, cls in enumerate(classes):
            subset = data[data[target] == cls]
            label = f'{target} = {cls}'

            if len(subset) < 2:
                continue

            if kde_samples and plot_type == 'kde':
                subset = subset.sample(n=min(kde_samples, len(subset)))

            if plot_type == 'scatter':
                sc = ax.scatter(subset[x_col], subset[y_col],
                              color=colors[i], label=label,
                              alpha=scatter_alpha, s=scatter_size,
                              edgecolor='none')
                handles.append(sc)
                labels.append(label)
                empty_plot = False

            elif plot_type == 'kde':
                try:
                    if subset[x_col].var() == 0 or subset[y_col].var() == 0:
                        continue

                    sns.kdeplot(x=subset[x_col], y=subset[y_col],
                               ax=ax, color=colors[i],
                               levels=kde_levels, alpha=kde_alpha,
                               thresh=0.1, warn_singular=warn_singular)
                    handles.append(Patch(color=colors[i], alpha=kde_alpha, label=label))
                    labels.append(label)
                    empty_plot = False
                except Exception as e:
                    pass

        if empty_plot:
            ax.text(0.5, 0.5, 'Недостаточно данных',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_facecolor('#f0f0f0')
        else:
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, alpha=0.3)

            if handles:
                ax.legend(handles=handles, labels=labels,
                         title=legend_title, loc=legend_location)

    # Построение графиков с фасетированием или без
    if facet_by is None:
        # Без фасетирования - оригинальная логика
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        base_size = 5 * figsize_multiplier
        fig, axes = plt.subplots(n_rows, n_cols,
                               figsize=(base_size*n_cols, base_size*n_rows),
                               squeeze=False)
        axes = axes.flatten()

        for idx, (x_col, y_col) in enumerate(pairs):
            plot_single(axes[idx], x_col, y_col, df)

        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

    else:
        # С фасетированием - создаем отдельную фигуру для каждой пары признаков
        facet_values = sorted(df[facet_by].unique())
        n_facets = len(facet_values)

        for x_col, y_col in pairs:
            fig, axes = plt.subplots((n_facets + facet_col_wrap - 1) // facet_col_wrap,
                                   facet_col_wrap,
                                   figsize=(5*facet_col_wrap*figsize_multiplier,
                                           4*((n_facets + facet_col_wrap - 1) // facet_col_wrap)*figsize_multiplier),
                                   squeeze=False)
            axes = axes.flatten()

            for i, facet_val in enumerate(facet_values):
                facet_data = df[df[facet_by] == facet_val]
                plot_single(axes[i], x_col, y_col, facet_data)
                axes[i].set_title(f"{facet_by} = {facet_val}")

            for j in range(n_facets, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle(f"Scatter plot for {x_col} vs {y_col} by {facet_by}", y=1.02)
            plt.tight_layout()

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    target_col: str,
    categorical_features: list = None,
    log_scale: bool = False,
    ncols: int = 5
):
    """
    Строит распределение категориальных признаков относительно бинарного целевого признака.

    Args:
        df (pd.DataFrame): Входной датафрейм с данными.
        target_col (str): Название целевого бинарного признака.
        categorical_features (list, optional): Список категориальных признаков для отображения.
            Если None, автоматически определяются все категориальные колонки в df (типы 'object' или 'category').
        log_scale (bool, optional): Если True, ось Y для графиков с абсолютными значениями будет в логарифмическом масштабе.
            По умолчанию False.
        ncols (int, optional): Количество столбцов в сетке графиков. По умолчанию 5.

    Raises:
        ValueError: Если целевой столбец отсутствует в df или не является бинарным.
        KeyError: Если указанные категориальные признаки отсутствуют в df.

    Example:
        >>> plot_categorical_distributions(df, 'Heart Attack Risk (Binary)', ['Gender', 'Smoking'])
        >>> plot_categorical_distributions(df, 'Heart Attack Risk (Binary)', ['Gender', 'Smoking'], log_scale=True, ncols=1)
    """

    # Проверка наличия целевого столбца
    if target_col not in df.columns:
        raise ValueError(f"Целевой столбец '{target_col}' отсутствует в датафрейме.")

    # Проверка, что целевой столбец бинарный
    unique_values = df[target_col].dropna().unique()
    if len(unique_values) != 2:
        raise ValueError(f"Целевой столбец '{target_col}' должен быть бинарным (иметь ровно 2 уникальных значения).")

    # Автоматическое определение категориальных признаков, если не переданы
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Исключаем целевой столбец из категориальных признаков
        categorical_features = [col for col in categorical_features if col != target_col]
    else:
        # Проверка наличия переданных категориальных признаков в датафрейме
        missing_cols = [col for col in categorical_features if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Следующие категориальные признаки отсутствуют в датафрейме: {missing_cols}")

    # Проверка, что есть признаки для отображения
    if not categorical_features:
        raise ValueError("Не найдено категориальных признаков для отображения.")

    # Вычисляем количество строк
    n_features = len(categorical_features)
    nrows = n_features * 2  # По 2 строки на признак (абсолютные значения и доли)
    ncols = min(ncols, n_features)  # Не больше, чем количество признаков
    
    # Создаем фигуру
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,  # Всегда возвращаем двумерный массив осей
        constrained_layout=True
    )
    fig.suptitle('Распределение категориальных признаков (абсолютные значения и доли)', fontsize=16)

    # Основной цикл для построения графиков
    for i, feature in enumerate(categorical_features):
        col = i % ncols
        row_abs = (i // ncols) * 2
        row_rel = row_abs + 1

        # График абсолютных значений
        sns.countplot(data=df, x=feature, hue=target_col, ax=axs[row_abs, col])
        axs[row_abs, col].set_title(f'Абсолютные значения: {feature}')
        axs[row_abs, col].tick_params(axis='x', rotation=45)
        axs[row_abs, col].legend(title=target_col, loc='upper right')
        axs[row_abs, col].grid(axis='y', linestyle='--', alpha=0.7)

        # Установка логарифмического масштаба для оси Y, если требуется
        if log_scale:
            axs[row_abs, col].set_yscale('log')
            axs[row_abs, col].set_ylabel('Количество (log scale)')
        else:
            axs[row_abs, col].set_ylabel('Количество')

        # График долей
        grouped = (df
                  .groupby([feature, target_col])
                  .size()
                  .unstack(fill_value=0)
                  .apply(lambda x: x / x.sum(), axis=1))
        grouped.plot(kind='bar', stacked=True, ax=axs[row_rel, col], colormap='viridis')
        axs[row_rel, col].set_title(f'Доли: {feature}')
        axs[row_rel, col].tick_params(axis='x', rotation=45)
        axs[row_rel, col].set_ylabel('Доля')
        axs[row_rel, col].legend(title=target_col, loc='upper right')
        axs[row_rel, col].grid(axis='y', linestyle='--', alpha=0.7)

    # Удаление пустых подграфиков
    for row in range(nrows):
        for col in range(ncols):
            feature_index = (row // 2) * ncols + col
            if feature_index >= len(categorical_features):
                fig.delaxes(axs[row, col])

    plt.show()


def plot_violin_combinations(
    df: pd.DataFrame,
    x_column: str = 'Age',
    y_features: list = None,
    hue_features: list = None,
    max_classes: int = 10,
    figsize_per_plot: tuple = (5, 5),
    numeric_to_categorical: bool = True,
    log_scale_x: bool = False
):
    """
    Строит комбинации violinplot для заданных/автоматически определённых категориальных признаков.

    Args:
    -----------
        df : pd.DataFrame
            Входной датафрейм с данными.
        x_column : str, optional (default='Age')
            Имя непрерывного числового признака для оси X (например, возраст).
        y_features : list, optional (default=None)
            Список категориальных признаков для оси Y.
            Если None, определяются автоматически:
            - Объектные столбцы с 2-10 уникальными значениями.
            Если numeric_to_categorical=True, числовые столбцы будут преобразованы в категориальные.
        hue_features : list, optional (default=None)
            Список бинарных категориальных признаков для разделения (hue).
            Если None, определяются автоматически:
            - Объектные/числовые столбцы с ровно 2 уникальными значениями.
        max_classes : int, optional (default=10)
            Максимальное количество уникальных значений для признаков на оси Y (по умолчанию до 10).
        figsize_per_plot : tuple, optional (default=(5, 5))
            Размер каждого подграфика в дюймах (width, height).
        numeric_to_categorical : bool, optional (default=True)
            Если True, числовые признаки в y_features будут преобразованы в категориальные.
        log_scale_x : bool, optional (default=False)
            Если True, ось X будет в логарифмическом масштабе.

    Returns:
    --------
    None
        Выводит графики и/или сообщения об ошибках.

    Example:
    --------
        >>> import pandas as pd
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Загрузка тестовых данных
        >>> df = sns.load_dataset('titanic')
        >>>
        >>> # Пример использования функции
        >>> plot_violin_combinations(
        ...     df=df,
        ...     x_column='age',
        ...     y_features=['class', 'embarked'],
        ...     hue_features=['sex', 'alive'],
        ...     figsize_per_plot=(6, 4),
        ...     log_scale_x=False
        ... )
        >>>
        >>> # Автоматический режим с логарифмической шкалой
        >>> plot_violin_combinations(
        ...     df=df,
        ...     x_column='fare',
        ...     log_scale_x=True
        ... )
    """

    # --- Шаг 1: Проверка корректности входных данных ---
    if x_column not in df.columns:
        raise ValueError(f"Столбец '{x_column}' не найден в датафрейме.")
    if not pd.api.types.is_numeric_dtype(df[x_column]):
        raise ValueError(f"Столбец '{x_column}' должен быть числовым.")

    # Создаем копию датафрейма для временных преобразований
    temp_df = df.copy()

    # --- Шаг 2: Автоматическое определение признаков ---
    if y_features is None:
        y_features = []
        for col in temp_df.select_dtypes(include=['object', 'category']).columns:
            if 2 <= temp_df[col].nunique(dropna=False) <= max_classes:
                y_features.append(col)

        if numeric_to_categorical:
            for col in temp_df.select_dtypes(include='number').columns:
                if col != x_column and 2 <= temp_df[col].nunique(dropna=False) <= max_classes:
                    # Сортируем числовые значения перед конвертацией в строки
                    unique_values = sorted(temp_df[col].dropna().unique())
                    temp_df[col] = temp_df[col].astype(str)
                    # Устанавливаем категориальный порядок для сортировки на графике
                    temp_df[col] = pd.Categorical(temp_df[col], categories=[str(v) for v in unique_values])
                    y_features.append(col)

        y_features = list(set(y_features))

    elif numeric_to_categorical:
        for col in y_features:
            if pd.api.types.is_numeric_dtype(temp_df[col]) and col != x_column:
                # Сортируем числовые значения перед конвертацией в строки
                unique_values = sorted(temp_df[col].dropna().unique())
                temp_df[col] = temp_df[col].astype(str)
                # Устанавливаем категориальный порядок для сортировки на графике
                temp_df[col] = pd.Categorical(temp_df[col], categories=[str(v) for v in unique_values])

    if hue_features is None:
        hue_features = []
        for col in temp_df.select_dtypes(include=['object', 'category']).columns:
            if temp_df[col].nunique(dropna=False) == 2:
                hue_features.append(col)
        for col in temp_df.select_dtypes(include='number').columns:
            if temp_df[col].nunique(dropna=False) == 2:
                hue_features.append(col)
        hue_features = list(set(hue_features))

    # --- Шаг 3: Проверка наличия подходящих признаков ---
    if not y_features:
        print("Не найдено подходящих категориальных признаков для оси Y (2-10 уникальных значений).")
        return
    if not hue_features:
        print("Не найдено бинарных признаков для разделения (hue).")
        return

    # --- Шаг 4: Генерация уникальных комбинаций (y, hue) ---
    combinations = []
    seen = set()
    for y_col in y_features:
        for hue_col in hue_features:
            if y_col != hue_col:
                key = tuple(sorted([y_col, hue_col]))
                if key not in seen:
                    combinations.append((y_col, hue_col))
                    seen.add(key)

    if not combinations:
        print("Не удалось сгенерировать комбинации (y, hue). Проверьте входные данные.")
        return

    # --- Шаг 5: Расчёт сетки подграфиков ---
    total_plots = len(combinations)
    ncols = min(3, total_plots)
    nrows = (total_plots + ncols - 1) // ncols

    # --- Шаг 6: Создание фигуры и подграфиков ---
    fig_width = ncols * figsize_per_plot[0]
    fig_height = nrows * figsize_per_plot[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if total_plots > 1 else [axes]

    # --- Шаг 7: Построение графиков ---
    for i, (y_col, hue_col) in enumerate(combinations):
        ax = axes[i]
        try:
            sns.violinplot(
                x=x_column,
                y=y_col,
                hue=hue_col,
                data=temp_df,
                split=True,
                palette='Set2',
                inner='quartile',
                ax=ax
            )

            # Применение логарифмического масштаба если нужно
            if log_scale_x:
                ax.set_xscale('log')
                ax.set_xlabel(f'{x_column} (log scale)')

            ax.set_title(f"{y_col} vs {hue_col}")
            ax.legend(title=hue_col, loc='upper right')
        except Exception as e:
            print(f"Ошибка при построении графика для {y_col} и {hue_col}: {str(e)}")
            ax.set_title(f"Ошибка: {y_col} vs {hue_col}")
            ax.text(0.5, 0.5, "Ошибка построения", ha='center', va='center')

    # --- Шаг 8: Удаление пустых подграфиков ---
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- Шаг 9: Финальная настройка и отображение ---
    plt.tight_layout()
    plt.show()


def plot_distributions(
    data,
    column,
    figsize=(12, 8),
    hist_bins=50,
    hist_color='steelblue',
    box_color='lightblue',
    title=None,
    xlabel=None,
    share_xlim=True
):
    """
    Рисует гистограмму и boxplot с выровненными осями X с использованием Seaborn.

    Args:
    ----------
        data : pandas.DataFrame
            Входной датафрейм.
        column : str
            Название столбца для анализа.
        figsize : tuple, optional
            Размер фигуры (ширина, высота). По умолчанию (12, 8).
        hist_bins : int, optional
            Количество бинов для гистограммы. По умолчанию 50.
        hist_color : str, optional
            Цвет гистограммы. По умолчанию 'steelblue'.
        box_color : str, optional
            Основной цвет boxplot. По умолчанию 'lightblue'.
        title : str, optional
            Общий заголовок. По умолчанию генерируется автоматически.
        xlabel : str, optional
            Подпись оси X. По умолчанию используется название столбца.
        share_xlim : bool, optional
            Синхронизировать пределы оси X. По умолчанию True.

    Examples:
    -----------------------
        >>> import pandas as pd
        >>> import numpy as np

        # Создаем тестовые данные
        >>> np.random.seed(42)
        >>> test_data = pd.DataFrame({
        ...     'Price': np.random.lognormal(mean=3, sigma=0.5, size=1000),
        ...     'Mileage': np.random.normal(loc=50000, scale=20000, size=1000)
        ... })

        # Базовый пример
        >>> plot_distributions(test_data, 'Price')

        # С кастомными параметрами
        >>> plot_distributions(
        ...     data=test_data,
        ...     column='Mileage',
        ...     figsize=(10, 6),
        ...     hist_bins=30,
        ...     hist_color='skyblue',
        ...     box_color='lightgreen',
        ...     title='Распределение пробега автомобилей',
        ...     xlabel='Пробег (км)'
        ... )
    """

    if column not in data.columns:
        raise ValueError(f"Столбец '{column}' не найден в датасете.")

    # Устанавливаем стиль Seaborn
    sns.set_style("whitegrid")

    # Создаем фигуру с двумя subplots
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=share_xlim,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Устанавливаем общий заголовок
    if title is None:
        title = f'Распределение {column}'
    fig.suptitle(title, fontsize=16, y=1.02)

    # Гистограмма с использованием Seaborn
    sns.histplot(
        data=data,
        x=column,
        bins=hist_bins,
        color=hist_color,
        edgecolor='white',
        ax=ax1,
        kde=False
    )

    ax1.set_ylabel('Количество', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Boxplot с использованием Seaborn
    sns.boxplot(
        data=data,
        x=column,
        color=box_color,
        ax=ax2,
        width=0.7,
        linewidth=1.5,
        flierprops={
            'marker': 'o',
            'markersize': 5,
            'markerfacecolor': 'none',
            'markeredgecolor': 'darkblue'
        }
    )

    # Устанавливаем цвет медианы вручную
    for line in ax2.lines:
        if line.get_linestyle() == '-':
            line.set_color('blue')
            line.set_linewidth(2)

    ax2.set_xlabel(column if xlabel is None else xlabel, fontsize=12)
    ax2.set_yticks([])
    ax2.grid(axis='x', alpha=0.3)

    # Убираем лишние границы
    for ax in [ax1, ax2]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


def phik_correlation_matrix(df, target_col=None, threshold=0.9, output_interval_cols=True, interval_cols=None):
    """Строит матрицу корреляции Фи-К (включая целевую переменную) и возвращает корреляции с целевой.

    Args:
        df (pd.DataFrame): DataFrame с данными для анализа
        target_col (str): Название столбца с целевой переменной
        threshold (float): Порог для выделения значимых корреляций (0.9 по умолчанию)
        output_interval_cols (bool): Возвращать ли список числовых непрерывных столбцов
        interval_cols (list): Список числовых непрерывных столбцов (если None, будет определен автоматически)

    Returns:
        tuple: (correlated_pairs, interval_cols, phi_k_with_target) где:
            - correlated_pairs: DataFrame с парами коррелирующих признаков
            - interval_cols: Список числовых непрерывных столбцов (если output_interval_cols=True)
            - phi_k_with_target: Series с корреляциями признаков с целевой переменной

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from phik import phik_matrix
        >>>
        >>> # Создаем тестовые данные
        >>> data = {
        ...     'price': [100, 200, 150, 300],  # Целевая переменная
        ...     'mileage': [50, 100, 75, 120],
        ...     'brand': ['A', 'B', 'A', 'C'],
        ...     'engine': [1.6, 2.0, 1.8, 2.5]
        ... }
        >>> df = pd.DataFrame(data)
        >>>
        >>> # Анализ корреляций с ручным заданием числовых столбцов
        >>> result = phik_correlation_matrix(df, target_col='price', threshold=0.3, interval_cols=['mileage', 'engine'])
        >>>
        >>> # Получаем результаты:
        >>> correlated_pairs = result[0]  # Пары коррелирующих признаков
        >>> interval_cols = result[1]     # Числовые непрерывные столбцы
        >>> phi_k_with_target = result[2] # Корреляции с ценой
        >>>
        >>> print("Корреляции с ценой:")
        >>> print(phi_k_with_target.sort_values(ascending=False))
    """

    # Определение числовых непрерывных столбцов (если не заданы вручную)
    if interval_cols is None:
        interval_cols = [
            col for col in df.select_dtypes(include=["number"]).columns
            if (df[col].nunique() > 50) or ((df[col] % 1 != 0).any())
        ]

    # Расчет полной матрицы корреляции (включая целевую переменную)
    phik_matrix = df.phik_matrix(interval_cols=interval_cols).round(2)

    # Получение корреляций с целевой переменной
    phi_k_with_target = None
    if target_col is not None and target_col in phik_matrix.columns:
        phi_k_with_target = phik_matrix[target_col].copy()
        # Удаляем корреляцию целевой с собой (всегда 1.0)
        phi_k_with_target.drop(target_col, inplace=True, errors='ignore')

    # Динамическое определение размера фигуры для подстройки размера ячеек
    num_cols = len(phik_matrix.columns)
    num_rows = len(phik_matrix.index)
    cell_size = 0.8  # Дюймов на ячейку
    figsize = (num_cols * cell_size, num_rows * cell_size)

    # Визуализация матрицы
    plot_correlation_matrix(
        phik_matrix.values,
        x_labels=phik_matrix.columns,
        y_labels=phik_matrix.index,
        vmin=0,
        vmax=1,
        color_map="Greens",
        title=r"Матрица корреляции $\phi_K$",
        fontsize_factor=1,
        figsize=figsize
    )
    plt.tight_layout()
    plt.show()

    # Фильтрация значимых корреляций (исключая целевую из пар)
    close_to_one = phik_matrix[phik_matrix >= threshold]
    close_to_one = close_to_one.where(
        np.triu(np.ones(close_to_one.shape), k=1).astype(bool)
    )

    # Удаление строк/столбцов с целевой переменной для анализа пар признаков
    if target_col is not None:
        close_to_one.drop(target_col, axis=0, inplace=True, errors='ignore')
        close_to_one.drop(target_col, axis=1, inplace=True, errors='ignore')

    # Преобразование в длинный формат
    close_to_one_stacked = close_to_one.stack().reset_index()
    close_to_one_stacked.columns = ["признак_1", "признак_2", "корреляция"]
    close_to_one_stacked = close_to_one_stacked.dropna(subset=["корреляция"])

    # Классификация корреляций
    def classify_correlation(corr):
        if corr >= 0.9: return "Очень высокая"
        elif corr >= 0.7: return "Высокая"
        elif corr >= 0.5: return "Заметная"
        elif corr >= 0.3: return "Умеренная"
        elif corr >= 0.1: return "Слабая"
        return "-"

    close_to_one_stacked["класс_корреляции"] = close_to_one_stacked["корреляция"].apply(
        classify_correlation
    )
    close_to_one_sorted = close_to_one_stacked.sort_values(
        by="корреляция", ascending=False
    ).reset_index(drop=True)

    if len(close_to_one_sorted) == 0 and threshold >= 0.9:
        print("\033[1mМультиколлинеарность между парами входных признаков отсутствует\033[0m")

    # Формирование результата
    result = [close_to_one_sorted]
    if output_interval_cols:
        result.append(interval_cols)
    if target_col is not None:
        result.append(phi_k_with_target)
    elif output_interval_cols:
        result.append(None)

    return tuple(result)


def vif(X, font_size=12):
    """Строит столбчатую диаграмму с коэффициентами инфляции дисперсии (VIF) для всех входных признаков.

    Args:
        X (pd.DataFrame): DataFrame с входными признаками для анализа.
        font_size (int): Размер шрифта для текстовых элементов графика (по умолчанию 12).

    Notes:
        - Коэффициент инфляции дисперсии (VIF) показывает степень мультиколлинеарности между признаками.
        - График отображается напрямую через matplotlib.

    Example:
        Пример использования функции:

        >>> import pandas as pd
        >>> from statsmodels.stats.outliers_influence import variance_inflation_factor
        >>> import statsmodels.api as sm
        >>>
        >>> # Создаем тестовый датафрейм
        >>> data = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [2, 4, 6, 8, 10],  # Полностью коррелирует с feature1
        ...     'feature3': [3, 6, 9, 12, 15]   # Частично коррелирует
        ... })
        >>>
        >>> # Вызываем функцию для анализа VIF
        >>> vif(data)
        >>>
        >>> # В результате будет показан график с VIF для каждого признака
        >>> # (feature2 будет иметь очень высокий VIF из-за полной корреляции с feature1)
    """
    # Кодируем категориальные признаки
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    # Добавляем константу для корректного расчета VIF
    X_with_const = sm.add_constant(X_encoded)

    # Вычисляем VIF для всех признаков, кроме константы (индексы начинаются с 1)
    vif = [variance_inflation_factor(X_with_const.values, i)
           for i in range(1, X_with_const.shape[1])]  # Исключаем константу (0-й столбец)

    # Построение графика с использованием исходных названий признаков (без константы)
    num_features = X_encoded.shape[1]
    fig_width = num_features * 1.2
    fig_height = 12

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(x=X_encoded.columns, y=vif)

    # Настройки графика
    ax.set_ylabel('VIF', fontsize=font_size)
    ax.set_xlabel('Входные признаки', fontsize=font_size)
    plt.title('Коэффициент инфляции дисперсии для входных признаков (VIF)', fontsize=font_size)

    # Метки на осях
    plt.xticks(rotation=90, ha='right', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Добавляем значения на столбцы (опционально)
    # ax.bar_label(ax.containers[0], fmt='%.2f', padding=3, fontsize=font_size)

    plt.tight_layout()
    plt.show()


def analyze_residuals(y_test, y_pred, units='ед. измерения', bins=30, figsize=(20, 6), 
                     title=None, lowess=False, lowess_frac=0.3):
    """Анализирует остатки модели и визуализирует результаты.

    Создаёт графики для анализа распределения остатков:
    - Гистограмма остатков с KDE-кривой
    - График остатков vs прогнозных значений (с опциональным LOWESS-сглаживанием)
    - График остатков vs номера наблюдения

    Args:
        y_test (Union[np.ndarray, list]): Вектор истинных значений.
        y_pred (Union[np.ndarray, list]): Вектор прогнозных значений.
        units (str, optional): Единицы измерения для осей. По умолчанию 'ед. измерения'.
        bins (int, optional): Количество бинов для гистограммы. По умолчанию 30.
        figsize (Tuple[float, float], optional): Размер фигуры (ширина, высота). 
            По умолчанию (20, 6).
        title (str, optional): Общий заголовок для графика. По умолчанию None.
        lowess (bool, optional): Включить LOWESS-сглаживание на графике остатков. 
            По умолчанию False.
        lowess_frac (float, optional): Параметр сглаживания для LOWESS (0-1). 
            По умолчанию 0.3.

    Returns:
        None: Выводит сетку графиков анализа остатков.

    Example:
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.model_selection import train_test_split
        >>> # Создаем синтетические данные
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 1)
        >>> y = 2 * X.squeeze() + np.random.normal(0, 0.1, 100)
        >>> # Разделяем на train/test
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> # Обучаем модель
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> # Анализируем остатки с LOWESS
        >>> analyze_residuals(y_test, y_pred, units='метры', title='Анализ остатков', 
        ...                  lowess=True, lowess_frac=0.25)
    """

    # Преобразование входных данных в numpy-массивы
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()

    # Проверка совпадения размерностей
    if len(y_test) != len(y_pred):
        raise ValueError("Длины y_test и y_pred должны совпадать")

    # Рассчитываем остатки
    error = y_test - y_pred

    # Создаем фигуру
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Добавляем общий заголовок
    if title:
        fig.suptitle(title)

    # Гистограмма остатков
    sns.histplot(error, bins=bins, kde=True, ax=axes[0])
    axes[0].axvline(
        x=0,
        color='r',
        linestyle='--',
        label='Нулевая линия'
    )
    axes[0].axvline(
        x=error.mean(),
        color='b',
        linestyle='--',
        label='Среднее'
    )
    axes[0].axvline(
        x=np.median(error),
        color='m',
        linestyle='-.',
        label='Медиана'
    )
    axes[0].axvline(
        x=error.mean() + error.std(),
        color='g',
        linestyle='--',
        label='Среднее ± std'
    )
    axes[0].axvline(
        x=error.mean() - error.std(),
        color='g',
        linestyle='--'
    )
    axes[0].set_title('Гистограмма остатков')
    axes[0].set_xlabel(f'Остатки ({units})')
    axes[0].set_ylabel('Частота')
    axes[0].legend()

    # Диаграмма рассеяния с возможностью LOWESS
    sns.scatterplot(x=y_pred, y=error, ax=axes[1], alpha=0.5)
    
    if lowess:
        # Применяем LOWESS-сглаживание
        lowess_sm = sm.nonparametric.lowess(error, y_pred, frac=lowess_frac)
        axes[1].plot(lowess_sm[:, 0], lowess_sm[:, 1], color='orange', 
                    linewidth=2, label=f'LOWESS (frac={lowess_frac})')
    
    axes[1].axhline(
        y=0,
        color='r',
        linestyle='--',
        label='Нулевая линия'
    )
    axes[1].axhline(
        y=error.mean(),
        color='b',
        linestyle='--',
        label='Среднее'
    )
    axes[1].axhline(
        y=np.median(error),
        color='m',
        linestyle='-.',
        label='Медиана'
    )
    axes[1].axhline(
        y=error.mean() + error.std(),
        color='g',
        linestyle='--',
        label='Среднее ± std'
    )
    axes[1].axhline(
        y=error.mean() - error.std(),
        color='g',
        linestyle='--'
    )
    axes[1].set_title('Диаграмма рассеяния прогнозов и остатков')
    axes[1].set_xlabel(f'Прогнозы ({units})')
    axes[1].set_ylabel(f'Остатки ({units})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_residuals(data, target, features):
    """
    Строит residplot для каждого указанного признака в датасете.

    Args:
    -----------
        data : pd.DataFrame
            Датасет с данными.
        target : str
            Имя целевой переменной.
        features : list of str
            Список входных признаков, для которых будут построены графики остатков.

    Examples:
    -----------------------
        >>> import pandas as pd
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Загрузка тестового датасета
        >>> tips = sns.load_dataset('tips')
        >>>
        >>> # Построение графиков остатков для нескольких признаков
        >>> plot_residuals(tips, target='total_bill', features=['tip', 'size', 'sex'])
    """
    n = len(features)
    cols = 3  # Максимум 3 графика в строке
    rows = (n + cols - 1) // cols  # Округление вверх

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Проверяем данные перед построением графика
        feature_data = data[feature]
        target_data = data[target]
        
        # Удаляем NaN значения
        mask = ~(feature_data.isna() | target_data.isna())
        clean_feature = feature_data[mask]
        clean_target = target_data[mask]
        
        # Проверяем, что осталось достаточно данных
        if len(clean_feature) < 10:
            ax.text(0.5, 0.5, f'Недостаточно данных\nдля {feature}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Остатки для признака {feature}')
            ax.set_xlabel(f'Признак {feature}')
            ax.set_ylabel('Остатки')
            continue
            
        # Проверяем дисперсию признака
        if clean_feature.std() == 0:
            ax.text(0.5, 0.5, f'Нулевая дисперсия\nдля {feature}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Остатки для признака {feature}')
            ax.set_xlabel(f'Признак {feature}')
            ax.set_ylabel('Остатки')
            continue
        
        try:
            # Пробуем построить график с LOWESS
            sns.residplot(
                x=clean_feature,
                y=clean_target,
                lowess=True,
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'black', 'linewidth': 2},
                ax=ax
            )
            ax.set_title(f'Остатки для признака {feature} (LOWESS)')
            
        except (FloatingPointError, ValueError, ZeroDivisionError, TypeError) as e:
            # Если LOWESS не работает, используем линейную регрессию
            try:
                print(f"LOWESS не сработал для {feature}: {str(e)[:100]}... Используем линейную регрессию.")
                sns.residplot(
                    x=clean_feature,
                    y=clean_target,
                    lowess=False,  # Отключаем LOWESS
                    scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red', 'linewidth': 2},
                    ax=ax
                )
                ax.set_title(f'Остатки для признака {feature} (линейная)')
            except Exception as e2:
                # Если и это не работает, показываем сообщение об ошибке
                ax.text(0.5, 0.5, f'Ошибка построения:\n{str(e2)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Ошибка для {feature}')
        
        ax.set_xlabel(f'Признак {feature}')
        ax.set_ylabel('Остатки')

    # Скрываем лишние пустые графики, если их больше, чем признаков
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_feature_importance(randomized_search, model_name_in_parentheses, catboost_features):
    """
    Визуализирует важность признаков обученной линейной модели в виде горизонтальной столбчатой диаграммы.
    Положительные коэффициенты отображаются красным цветом, отрицательные - синим.

    Args:
    ----------
        randomized_search : RandomizedSearchCV
            Обученный объект RandomizedSearchCV с лучшей моделью, содержащей шаги:
            - feature_union: для получения имен признаков
            - regressor: обученная линейная модель с коэффициентами
        model_name_in_parentheses : str
            Название модели, которое будет отображаться в скобках в заголовке графика
            (например: 'Ridge', 'Lasso', 'LinearRegression')
        catboost_features : List
            Признаки для CatBoostEncoder

    Return:
    -----------
    None
        Функция отображает график с помощью matplotlib.pyplot.show()

    Example:
    ---------------------
    >>> # Импорт необходимых библиотек
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.compose import ColumnTransformer
    >>> from category_encoders import CatBoostEncoder
    >>>
    >>> # Определение типов признаков
    >>> binary_features = ['is_active']
    >>> ohe_features = ['category']
    >>> numeric_features = ['age', 'income']
    >>> catboost_features = ['city']
    >>>
    >>> # Создание тестовых данных
    >>> data = {
    ...     'is_active': [1, 0, 1, 1],
    ...     'category': ['A', 'B', 'A', 'C'],
    ...     'age': [25, 32, 45, 28],
    ...     'income': [50000, 65000, 80000, 55000],
    ...     'city': ['Moscow', 'SPb', 'Moscow', 'Kazan']
    ... }
    >>> X_train = pd.DataFrame(data)
    >>> y_train = pd.Series([100, 150, 200, 120])
    >>>
    >>> # Создание пайплайна предобработки
    >>> ohe_pipe = Pipeline([
    ...     ('imputer', SimpleImputer(strategy='most_frequent')),
    ...     ('ohe', OneHotEncoder(drop='first')),
    ... ])
    >>>
    >>> numeric_pipe = Pipeline([
    ...     ('imputer', SimpleImputer(strategy='median')),
    ...     ('scaler', StandardScaler())
    ... ])
    >>>
    >>> preprocessor = ColumnTransformer([
    ...     ('ohe', ohe_pipe, binary_features + ohe_features),
    ...     ('numeric', numeric_pipe, numeric_features)
    ... ])
    >>>
    >>> # Пайплайн для CatBoostEncoder
    >>> catboost_pipe = Pipeline([
    ...     ('imputer', SimpleImputer(strategy='most_frequent')),
    ...     ('catboost', CatBoostEncoder())
    ... ])
    >>>
    >>> # Создание финального пайплайна с моделью
    >>> pipe_final = Pipeline([
    ...     ('feature_union', FeatureUnionWithCatBoost(
    ...         preprocessor=preprocessor,
    ...         catboost_pipe=catboost_pipe,
    ...         catboost_features=catboost_features
    ...     )),
    ...     ('regressor', Ridge())
    ... ])
    >>>
    >>> # Параметры для RandomizedSearchCV
    >>> param_grid = {
    ...     'regressor__alpha': [0.1, 1.0, 10.0]
    ... }
    >>>
    >>> # Обучение RandomizedSearchCV
    >>> randomized_search = RandomizedSearchCV(
    ...     pipe_final,
    ...     param_grid,
    ...     cv=3,
    ...     n_iter=3,
    ...     random_state=42
    ... )
    >>> randomized_search.fit(X_train, y_train)
    >>>
    >>> # Визуализация важности признаков
    >>> plot_feature_importance(
    ...     randomized_search=randomized_search,
    ...     model_name_in_parentheses='Ridge',
    ...     catboost_features=catboost_features
    ... )
    """
    
    # 1. Получаем обученную модель
    best_model = randomized_search.best_estimator_
    trained_model = best_model.named_steps['regressor']

    # 2. Извлекаем коэффициенты
    coefficients = trained_model.coef_

    # 3. Получаем имена признаков из всех частей пайплайна
    # Получаем имена признаков из preprocessor (OHE и числовые признаки)
    feature_union = best_model.named_steps['feature_union']
    preprocessor = feature_union.preprocessor
    
    # Для sklearn 0.24.1 используем старый способ получения имен признаков
    ohe_transformer = preprocessor.named_transformers_['ohe']
    ohe = ohe_transformer.named_steps['ohe']
    
    # Получаем имена категориальных признаков после OHE
    ohe_feature_names = ohe.get_feature_names_out(input_features=binary_features + ohe_features)
    
    # Числовые признаки остаются как есть
    numeric_feature_names = numeric_features
    
    # Объединяем имена признаков из preprocessor
    preprocessor_feature_names = list(ohe_feature_names) + numeric_feature_names

    # Получаем имена признаков из CatBoostEncoder (добавляем префикс)
    catboost_feature_names = [f'catboost_{feature}' for feature in catboost_features]

    # Объединяем все имена признаков
    feature_names = preprocessor_feature_names + catboost_feature_names

    # 4. Создаем DataFrame для удобства
    coefficients_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # 5. Сортируем по абсолютному значению коэффициентов
    sorted_coefficients = coefficients_df.sort_values(
        by='Coefficient',
        key=lambda x: abs(x),
        ascending=True
    ).reset_index(drop=True)

    # 6. Визуализация
    plt.figure(figsize=(12, 8))

    # Создаем список цветов
    colors = ['lightcoral' if coeff > 0 else 'skyblue' for coeff in sorted_coefficients['Coefficient']]

    # Создаем горизонтальную столбчатую диаграмму
    bars = plt.barh(
        y=sorted_coefficients['Feature'],
        width=sorted_coefficients['Coefficient'],
        color=colors
    )

    # Настройки графика
    plt.xlabel('Значение коэффициента', fontsize=12)
    plt.ylabel('Признак', fontsize=12)
    plt.title(f'Важность признаков в линейной регрессии ({model_name_in_parentheses})',
             fontsize=14, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Добавляем значения коэффициентов
    for bar in bars:
        width = bar.get_width()
        plt.text(
            x=width + 0.01 * np.sign(width),
            y=bar.get_y() + bar.get_height()/2,
            s=f'{width:.2f}',
            va='center',
            ha='center' if width == 0 else 'left' if width > 0 else 'right',
            fontsize=8
        )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()