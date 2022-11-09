# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

COPY ./entrypoint.sh /
EXPOSE 8000
ENTRYPOINT ["sh", "/entrypoint.sh"]
