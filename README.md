# XGBoost Landmark Classification

Un proyecto de machine learning que utiliza XGBoost para clasificación de puntos de referencia (landmarks) a partir de datos de segmentos. El proyecto procesa datos en formato `.npy` y entrena un modelo de clasificación.

## Descripción

Este proyecto incluye:
- **Entrenamiento del modelo**: Script para entrenar un clasificador XGBoost con datos de landmarks
- **Predicción**: Script principal para realizar predicciones en nuevos dados
- **Preprocesamiento**: Funciones de utilidad para limpiar y aumentar los datos
- **Dockerización**: Soporte completo para ejecutar el proyecto en contenedores Docker

## Requisitos

- Python 3.x
- Las dependencias se encuentran en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone <URL_DEL_REPOSITORIO>
cd challenge_alejandro_diaz_montes_de_oca
```

2. Crear un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Entrenar el modelo

```bash
python train_model.py
```

Este script procesará los datos de entrenamiento y generará los artefactos del modelo en la carpeta `model_artifacts/`.

### Ejecutar predicciones

```bash
python main.py <input_dir> <output_dir>
```

O usando variables de entorno (para Docker):
```bash
export INPUT=<ruta_input>
export OUTPUT=<ruta_output>
python main.py
```

## Estructura del Proyecto

```
├── main.py                 # Script principal de predicción
├── train_model.py          # Script de entrenamiento del modelo
├── utils.py                # Funciones de utilidad
├── requirements.txt        # Dependencias del proyecto
├── Dockerfile              # Configuración para ejecutar en Docker
├── README.md               # Este archivo
├── .gitignore              # Archivos a ignorar en Git
├── data/                   # Datos de entrenamiento y testing (archivos .npy)
├── model_artifacts/        # Artefactos del modelo entrenado
└── output/                 # Resultados y salidas del modelo
```

## Docker

Para ejecutar el proyecto en Docker:

```bash
docker build -t xgboost-classifier .
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output xgboost-classifier
```

## Autor

Alejandro Díaz Montes de Oca

## Licencia

Especificar la licencia si corresponde
