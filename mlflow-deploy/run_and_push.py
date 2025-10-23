import subprocess
import sys
import os

# --- Configuración ---
COMMIT_MESSAGE = "feat: Ejecucion de CI/CD local y push a GitHub Actions"
# Cambia 'main' por la rama que estés usando (ej. 'develop')
BRANCH_NAME = "main" 
PYTHON_EXE = sys.executable

def run_command(command, cwd=None):
    """Ejecuta un comando de sistema y maneja errores."""

    display_command = list(command)
    print(f"\n>>>> Ejecutando: {' '.join(command)}")
    try:
        # Ejecuta el comando, capturando la salida y error
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,  # Lanza una excepción si el comando falla
            text=True,
            capture_output=True,
            encoding='cp1252'
        )
        print("Salida del comando:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: El comando falló con código de salida {e.returncode}.")
        print("--- Salida de Error ---")
        print(e.stderr)
        print("------------------------")
        sys.exit(1) # Termina el script si un paso falla
    except FileNotFoundError:
        print(f"\n❌ ERROR: Comando no encontrado. Asegúrate de tener 'git', 'pip' y 'make' instalados y en el PATH.")
        sys.exit(1)

def run_ci_cd_and_push():
    """Ejecuta el pipeline CI/CD local y sube los cambios a GitHub."""
    
    # El directorio actual es donde se debe ejecutar 'pip' y 'make'
    project_root = os.path.dirname(os.path.abspath(__file__))

    print("--- PASO 1: INSTALAR DEPENDENCIAS ---")
    # Comando: pip install -r requirements.txt
    run_command(["pip", "install", "-r", "requirements.txt"], cwd=project_root)

    print("\n--- PASO 2: EJECUTAR PIPELINE CI LOCAL (make ci) ---")
    # 1. Entrenar y validar el modelo de Regresión Lineal
    print("\n>>>> Iniciando experimento LINEAL...")
    run_command([PYTHON_EXE, "src/train_LinearR.py"], cwd=project_root)
    run_command([PYTHON_EXE, "src/validate.py"], cwd=project_root)
    
    # 2. Entrenar y validar el modelo de Regresión Logística
    print("\n>>>> Iniciando experimento LOGÍSTICO...")
    run_command([PYTHON_EXE, "src/train_Logistic.py"], cwd=project_root)
    run_command([PYTHON_EXE, "src/validate.py"], cwd=project_root)

    print("\n--- PASO 3: GIT COMMIT Y PUSH ---")
    
    # Comando: git add .
    run_command(["git", "add", "."], cwd=project_root)
    
    # Comando: git commit -m "..."
    if run_command(["git", "commit", "-m", COMMIT_MESSAGE], cwd=project_root):
        print(f"✅ Commit creado con éxito.")
    
    # Comando: git push origin [BRANCH_NAME]
    run_command(["git", "push", "origin", BRANCH_NAME], cwd=project_root)
    
    print("\n=======================================================")
    print("🚀 ¡Proceso finalizado con éxito! El push a GitHub")
    print("   activará tu pipeline de GitHub Actions (CI/CD).")
    print("=======================================================")


if __name__ == "__main__":
    # Nos aseguramos de que el script se ejecute desde el directorio correcto (la raíz del repositorio)
    # Debes ejecutar este script un nivel arriba de la carpeta 'TRADINGTOP'.
    # Si lo ejecutas dentro de la carpeta 'mlflow-deploy', deberías ajustar el ROOT_DIR.
    # Por defecto, asumimos que este script está en la raíz del repositorio.
    run_ci_cd_and_push()