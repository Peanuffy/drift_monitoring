import pandas as pd
import mlflow

LOG_FILE = "../serving/requests_log.csv"
MODEL_NAME = "DiamondsPriceModel"

mlflow.set_tracking_uri("file:../mlruns")
client = mlflow.MlflowClient()

# === Загрузка логов ===
df = pd.read_csv(LOG_FILE)

prod = df[df["model_version"] == "Production"]
stag = df[df["model_version"] == "Staging"]

if len(prod) < 10 or len(stag) < 10:
    raise ValueError("Недостаточно данных для A/B теста")

# === Метрики ===
metrics = {
    "prod_mean": prod["prediction"].mean(),
    "stag_mean": stag["prediction"].mean(),
    "prod_std": prod["prediction"].std(),
    "stag_std": stag["prediction"].std(),
}

print("📊 A/B метрики:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# === Логика победителя ===
# Пример: модель стабильнее (меньше std)
if metrics["stag_std"] < metrics["prod_std"]:
    print("🚀 Staging модель лучше → переводим в Production")

    # Найти версию Staging
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not versions:
        raise RuntimeError("Нет модели в Staging")

    staging_version = versions[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✅ Версия {staging_version} переведена в Production")

else:
    print("❌ Production модель остаётся лучшей")


