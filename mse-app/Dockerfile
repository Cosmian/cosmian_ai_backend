FROM python:3.11-slim-buster

WORKDIR /usr/src/app

ENV MODULEPATH /usr/src/app

RUN apt-get update && \
    apt-get install -y build-essential python3-dev

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./mse_src /usr/src/app
COPY ./secrets.json .

CMD gunicorn --bind 0.0.0.0:5000 wsgi:app