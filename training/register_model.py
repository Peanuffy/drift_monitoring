import mlflow
import mlflow.sklearn
from pycaret.regression import load_model
from pathlib import Path

# === Конфигурация ===
TRACKING_URI = "file:../mlruns"   # MLflow локальная папка
REGISTERED_MODEL_NAME = "DiamondsPriceModel"
LOCAL_MODEL_PATH = "diamonds_model"  # путь к локальной модели PyCaret

# === Подключаем MLflow ===
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.MlflowClient()

# === Загружаем локальную PyCaret модель ===
final_model = load_model(LOCAL_MODEL_PATH)

# === Проверяем, существует ли уже зарегистрированная модель ===
try:
    client.get_registered_model(REGISTERED_MODEL_NAME)
    print(f"Model '{REGISTERED_MODEL_NAME}' уже зарегистрирована.")
except mlflow.exceptions.MlflowException:
    print(f"Model '{REGISTERED_MODEL_NAME}' не найдена. Создаём новую регистрацию.")

# === Логирование и регистрация модели ===
with mlflow.start_run(run_name="register_model"):
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

print(f"Модель '{REGISTERED_MODEL_NAME}' успешно зарегистрирована и добавлена в MLflow Model Registry (Staging).")

# === Опционально: вывести версии модели ===
latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
for v in latest_versions:
    print(f"Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")
