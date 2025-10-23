# --- Variables ---
# Define el puerto de servicio
PORT = 5001

# Define el nombre del modelo (simula el uso de MLflow Model Registry)
MODEL_NAME = best_model

# --- Targets de Utilidad ---

install:
	@echo "⚙️ Instalando dependencias..."
	pip install -r requirements.txt

clean:
	@echo "🗑️ Limpiando artefactos de MLflow..."
	rm -rf mlruns/
	
# --- Targets de Entrenamiento y Validación ---

train-linear:
	@echo "🚀 Ejecutando entrenamiento del modelo de Regresión Lineal..."
	python src/train_linear.py

train-logistic:
	@echo "🚀 Ejecutando entrenamiento del modelo de Regresión Logística..."
	python src/train_logistic.py

validate:
	@echo "✅ Ejecutando validación del último Run de MLflow..."
	python src/validate.py

run-all: train-linear validate train-logistic validate

# --- Targets de Servicio/Despliegue ---

# 1. Opción simple (Servicio local del último run - Requiere saber la ruta)
# Esto es difícil de automatizar sin scripting, pero es el comando directo.
# Si quieres correr un modelo manualmente: make serve_manual path=/path/to/run/artifacts/model
serve_manual:
	@echo "🌐 Sirviendo el modelo desde la ruta especificada..."
	mlflow models serve -m $(path) -p $(PORT) --no-conda

# 2. Opción Avanzada (Simulando Model Registry)
# En un ambiente real de MLOps, el 'validate.py' promovería el mejor modelo
# a un nombre específico en el Model Registry, y servirías por nombre.
# NOTA: Para que esto funcione, necesitarás un servidor MLflow remoto o 
# configurar el Model Registry en local.
serve_registry:
	@echo "🌐 Sirviendo la última versión del modelo registrado 'best_model'..."
	mlflow models serve -m models:/"$(MODEL_NAME)"/latest -p $(PORT) --no-conda