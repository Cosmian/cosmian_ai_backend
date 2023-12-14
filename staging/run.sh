#!/bin/bash

export $(cat config.env | xargs)
cd app && gunicorn --bind 0.0.0.0:5001 wsgi:app
