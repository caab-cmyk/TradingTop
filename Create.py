import os
from pathlib import Path

# 1. Define el directorio raíz del proyecto
ROOT_DIR = r"C:\Users\thene\OneDrive - universidadean.edu.co\VS Code\PythonFinanzas\TradingTop\mlflow-deploy"

# 2. Define la lista de todos los archivos y carpetas a crear
# Las rutas se definen como tuplas (carpeta, archivo) o (carpeta,)
project_structure = [
    # Directorios
    (ROOT_DIR, "src"),
    (ROOT_DIR, "mlruns"),
    (ROOT_DIR, ".github/workflows"),
    
    # Archivos dentro de src/
    (f"{ROOT_DIR}/src", "train.py"),
    (f"{ROOT_DIR}/src", "validate.py"),
    (f"{ROOT_DIR}/src", "serve.py"), # (opcional)
    
    # Archivos en el directorio raíz
    
    (ROOT_DIR, "requirements.txt"),
    
    # Archivo de CI/CD
    (f"{ROOT_DIR}/.github/workflows", "ci-cd.yml"),
]

def create_project_structure(structure):
    """Crea directorios y archivos basándose en la lista de la estructura del proyecto."""
    print(f"Iniciando la creación de la estructura para el proyecto '{ROOT_DIR}'...")
    
    for path_tuple in structure:
        # Usamos Path para manejar rutas de manera robusta en cualquier sistema operativo
        
        if len(path_tuple) == 1:
            # Caso: solo es un directorio (ya manejado implícitamente abajo, pero útil para claridad)
            path = Path(path_tuple[0])
        elif len(path_tuple) == 2:
            # Caso: path_tuple = (directorio_base, nombre_elemento)
            path = Path(path_tuple[0]) / path_tuple[1]
        else:
            continue

        if "." not in path.name:
            # Es un directorio: lo creamos junto con sus padres (similar a 'mkdir -p')
            os.makedirs(path, exist_ok=True)
            print(f"  ✅ Directorio creado: {path}")
        else:
            # Es un archivo: nos aseguramos de que su directorio exista y luego creamos el archivo vacío
            # path.parent es el directorio donde debe ir el archivo
            os.makedirs(path.parent, exist_ok=True) 
            path.touch(exist_ok=True) # Crea el archivo si no existe, lo deja si existe
            print(f"  📝 Archivo creado: {path}")

if __name__ == "__main__":
    create_project_structure(project_structure)
    print(f"\n¡Estructura de proyecto '{ROOT_DIR}' creada exitosamente!")