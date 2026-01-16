import mlflow
import mlflow.sklearn
from pycaret.regression import load_model
from mlflow.exceptions import MlflowException

# === Настройки ===
mlflow.set_tracking_uri("file:../mlruns")
MODEL_NAME = "DiamondsPriceModel"
LOCAL_MODEL_PATH = "diamonds_drift_model"

client = mlflow.MlflowClient()

all_versions = client.get_latest_versions(MODEL_NAME, stages=None)

# Находим **максимальную версию** — это последняя зарегистрированная
latest_version = max([v.version for v in all_versions])

# Переводим её сразу в Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Staging",
    archive_existing_versions=False  # оставляем старые версии как есть
)

print(f"Модель зарегистрирована и переведена в Staging. Version: {latest_version}")


print(f"Модель '{MODEL_NAME}' успешно зарегистрирована и добавлена в MLflow Model Registry (Staging).")

# === Опционально: вывести версии модели ===
latest_versions = client.get_latest_versions(MODEL_NAME)
for v in latest_versions:
    print(f"Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")