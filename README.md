# Instruccinoes de ejecución
1. Activar el entorno virtual: .venv/scripts/activate 
2. En la terminal escribir el comando: python -m app.app
3. Hacer requests

# Endpoints
1. health/: Endpoint para ver el estado de la API
2. info/: Endpoint con información de los parámetros del modelo
3. predict/: Endpoint con servicio de predicción. Se le debe proporcionar una solicitud en formato JSON con el siguiente formato:

    {
        "features": [valor1, valor2, valor3, valor4]
    }


donde cada valor es un número flotante mayor a 0 y menor a 10