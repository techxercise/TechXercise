# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY . .

EXPOSE 8080
CMD ["gunicorn", "app:app", "-b", ":8080", "--timeout", "300"]