#!/usr/bin/env bash

# Start the Telegram bot in the background
python manage.py run_bot &

# Start the Django web server in the foreground
exec gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000}
