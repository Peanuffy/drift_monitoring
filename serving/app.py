from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd
import random
from datetime import datetime
import os

app = Flask(__name__)

# ===== Настройки =====
mlflow.set_tracking_uri("file:../mlruns")
MODEL_NAME = "DiamondsPriceModel"
LOG_FILE = "requests_log.csv"
PROD_RATIO = 0.7  # доля трафика на Production

client = mlflow.MlflowClient()

# ===== Загружаем модели =====
def load_latest_model(stage):
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not versions:
        return None
    model_uri = f"models:/{MODEL_NAME}/{versions[-1].current_stage}"
    return mlflow.pyfunc.load_model(model_uri)

prod_model = load_latest_model("Production")
stag_model = load_latest_model("Staging")

if stag_model is None:
    raise Exception("Staging модель не найдена!")

@app.route("/")
def home():
    return "Flask сервер работает!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Неверный формат данных"}), 400

    df = pd.DataFrame([data])

    # Выбираем модель для запроса
    if prod_model and random.random() < PROD_RATIO:
        model = prod_model
        model_version = "Production"
    else:
        model = stag_model
        model_version = "Staging"

    prediction = model.predict(df)[0]

    # Логирование запроса
    log_entry = {"timestamp": datetime.now(), "model_version": model_version, **data, "prediction": prediction}

    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log_entry])

    df_log.to_csv(LOG_FILE, index=False)

    return jsonify({"prediction": prediction, "model_version": model_version})

if __name__ == "__main__":
    # host="0.0.0.0" позволяет подключаться с любого IP на Windows
    app.run(debug=True, host="0.0.0.0", port=5001)
