FROM python:3.11-slim-buster

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y build-essential python3-dev

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./src /usr/src/app

ENV SUMMARY_MODEL facebook/bart-large-cnn
ENV TRANSLATION_MODEL facebook/nllb-200-distilled-600M

CMD gunicorn --bind 0.0.0.0:5000 --timeout 600 wsgi:app
