FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Esta línea sólo documenta el puerto, no hardcodea nada:
EXPOSE 8000

# Con shell form podemos usar la variable $PORT que Render inyecta
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
