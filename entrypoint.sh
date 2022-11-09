#!/bin/sh

cd handwrittin
python manage.py migrate --no-input
python manage.py collectstatic --no-input

gunicorn handwrittin.wsgi:application --bind 0.0.0.0:8000
