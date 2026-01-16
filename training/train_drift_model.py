import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Загружаем исходные данные
df = pd.read_csv("../data/diamonds.csv")

# Разделяем на train/test
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Создаём "дрейфовые" данные
drift_test = test.copy()

# Например, увеличим carat и изменим распределение cut
drift_test["carat"] = drift_test["carat"] * np.random.uniform(1.05, 1.25, size=len(drift_test))
cut_options = ["Ideal", "Premium", "Good", "Very Good"]
drift_test["cut"] = np.random.choice(cut_options, size=len(drift_test), p=[0.1, 0.5, 0.3, 0.1])

drift_test.to_csv("diamonds_drift.csv", index=False)
print("Drift dataset created:", drift_test.shape)

from pycaret.regression import setup, compare_models, finalize_model, save_model

import mlflow
mlflow.set_tracking_uri("file:../mlruns")

# Загружаем дрейфовые данные
df_drift = pd.read_csv("diamonds_drift.csv")

# PyCaret setup
s = setup(
    data=df_drift,
    target="price",
    session_id=42,
    log_experiment=True,
    experiment_name="diamonds_automl_drift",
)

# Сравниваем модели
best_model = compare_models()

# Финализируем
final_model = finalize_model(best_model)

# Сохраняем локально
save_model(final_model, "diamonds_drift_model")
