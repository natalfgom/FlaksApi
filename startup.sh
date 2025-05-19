#!/bin/bash

# Create and activate virtual environment
python -m venv antenv
source antenv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=production

# Start Gunicorn
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app 