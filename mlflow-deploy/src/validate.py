import mlflow
import json
import os
mlflow.set_tracking_uri("mlflow-deploy\mlruns")
experiment = mlflow.get_experiment_by_name("ci-cd-mlflow-local")
runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
# Último run
mse = runs.iloc[0]["metrics.mse"]
print(f"Validando MSE: {mse}")
if mse > 3000:
    raise ValueError("❌ MSE demasiado alto, no se puede promover el modelo.")
else:
    print(" MSE aceptable. Modelo listo para promoción.")
