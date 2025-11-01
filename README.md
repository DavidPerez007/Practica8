# Laboratorio 8: Random Forest y Despliegue de API

## Integrantes del Equipo
- **David Leobardo Pérez Cruz** (DavidPerez007)
- **Imanol Mendoza Sáenz de Buruaga** (ImanolMSB)
- **Emil Ehecatl Sánchez Olsen** (Emilehecatlsanchez)
- **Erick José Fabián Sandoval** (erick0x)

## Repositorio
https://github.com/DavidPerez007/PredictionService.git

## Fecha de Entrega
31/10/2025

---

## Descripción del Proyecto

Este proyecto implementa un modelo de **Random Forest desde cero** para clasificación, realiza una comparativa con la implementación de scikit-learn, y despliega el modelo entrenado como una **API REST funcional** utilizando FastAPI en Render.

## Estructura del Proyecto

PredictionService/

├── model/

│ ├── modelo.pkl

│ └── rf_custom.py

├── app/

│ ├── main.py

│ └── requirements.txt

├── notebooks/

│ └── Training.ipynb

├── render.yaml

└── README.md


## Implementación Técnica

### Elemento 1: Random Forest Personalizado
- **Clase RandomForest**: Implementación desde cero con:
  - Muestreo bootstrap con reemplazo
  - Entrenamiento de múltiples árboles de decisión
  - Agregación por voto mayoritario
  - Soporte para parámetros: n_estimators, max_depth, max_features, random_state

### Elemento 2: Comparativa con Scikit-learn
- **Preprocesamiento**: Limpieza de datos atípicos usando KNNImputer
- **Métricas**: Accuracy, matriz de confusión, reporte de clasificación
- **Optimización**: Búsqueda de hiperparámetros con GridSearchCV

### Elemento 3: API y Despliegue
- **Framework**: FastAPI
- **Endpoints**:
  - GET /health: Verificación del estado del servicio
  - GET /info: Información del modelo y equipo
  - POST /predict: Endpoint de predicción

## Instalación y Uso Local

### Prerrequisitos
- Python 3.10+
- Dependencias listadas en app/requirements.txt

### Instalación

- git clone https://github.com/DavidPerez007/PredictionService.git

- cd PredictionService

- pip install -r app/requirements.txt


### Ejecución Local

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


La API estará disponible en: http://localhost:8000

## Endpoints de la API

### 1. Health Check

https://predictionservice.onrender.com/health

Respuesta:


{"status":"ok"}

### 2. Información del Modelo

https://predictionservice.onrender.com/info

Respuesta:
{
  "max_depth": 2,
  "model": "RandomForest",
  "n_estimators": 100,
  "random_state": 17,
  "team": "Feliz Navidad"
}


### 3. Predicción

Mandando esto con python 

{
"features": [5.1, 3.5, 1.4, 0.2]
}

Se obtiene:

 {'prediction': 'Setosa'}



## Despliegue en Render

El servicio está desplegado usando Render con la configuración en render.yaml:

services:
  - type: web
    name: random-forest-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn app.main:app --host 0.0.0.0 --port 10000"

URL del servicio desplegado: https://predictionservice.onrender.com

## Resultados y Métricas

### Preprocesamiento
- Dataset: Iris con 125 muestras
- Limpieza: Eliminación de valores atípicos usando KNNImputer
- Características: 4 dimensiones (sepal length, sepal width, petal length, petal width)

### Rendimiento del Modelo
- Random Forest Personalizado: 96% accuracy en entrenamiento
- Scikit-learn Random Forest: 100% accuracy en entrenamiento
- Mejores Parámetros: n_estimators=50, max_depth=None

## Características Técnicas

### Implementación Personalizada
- Bootstrap sampling con reemplazo
- Soporte para max_features como 'sqrt' o 'log2'
- Votación mayoritaria para clasificación

### API Features
- Validación de entrada de datos
- Respuestas JSON estandarizadas