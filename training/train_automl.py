import pandas as pd
from pycaret.regression import *
import mlflow
import mlflow.sklearn

# === 1. Load data ===
df = pd.read_csv("../data/diamonds.csv")

# === 2. PyCaret setup (БЕЗ логирования) ===
s = setup(
    data=df,
    target="price",
    session_id=42,
    log_experiment=False,
    system_log=False
)

# === 3. AutoML ===
best_model = compare_models()

# === 4. Final model ===
final_model = finalize_model(best_model)

# === 5. MLflow logging (ЯВНО) ===
mlflow.set_tracking_uri("file:../mlruns")

with mlflow.start_run(run_name="diamonds_automl_run"):
    mlflow.log_param("model_class", type(final_model).__name__)
    mlflow.sklearn.log_model(
        final_model,
        artifact_path="model",
        registered_model_name="DiamondsPriceModel"
    )

# === 6. Save locally ===
save_model(final_model, "diamonds_model")
