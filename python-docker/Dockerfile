# syntax=docker/dockerfile:1

FROM python:3.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -U numpy

WORKDIR /app

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]