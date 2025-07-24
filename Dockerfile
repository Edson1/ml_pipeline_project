# for FastAPI application with AWS SageMaker endpoint
# Imagen base oficial de Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto
COPY main.py . 
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa FastAPI
EXPOSE 8000

# Comando para ejecutar el servidor univcorn para fastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
