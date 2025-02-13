FROM python:3.13-slim

COPY weather_dashboard.py weather_dashboard.py
COPY requirements_dashboard.txt requirements_dashboard.txt

RUN pip install -r requirements_dashboard.txt

EXPOSE 8050

CMD ["python", "weather_dashboard.py"]
