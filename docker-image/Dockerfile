FROM python:3.11-slim-buster

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y build-essential python3-dev

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./src /usr/src/app

ENV MODEL_NAME facebook/bart-large-cnn
ENV KMS_URL https://kms.url
ENV KMS_API_KEY XXXX

CMD gunicorn --bind 0.0.0.0:5000 wsgi:app
