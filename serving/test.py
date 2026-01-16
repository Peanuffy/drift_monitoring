import mlflow

mlflow.set_tracking_uri("file:../mlruns")
client = mlflow.MlflowClient()

# Список всех зарегистрированных моделей (вместо list_registered_models)
for rm in client.search_registered_models():
    print("Registered model:", rm.name)

# Все версии DiamondsPriceModel
MODEL_NAME = "DiamondsPriceModel"
for v in client.get_latest_versions(MODEL_NAME):
    print(f"Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")
