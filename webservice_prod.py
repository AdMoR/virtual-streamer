#!/usr/bin/env python3
"""
Production WSGI server for the webservice using gunicorn
"""
import os
import multiprocessing

# Gunicorn config
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
# Number of workers based on CPU count, max 4 as before. Uvicorn workers handle concurrency differently.
# Start with a reasonable number, maybe fewer than sync workers depending on workload.
workers = 1
timeout = 300  # 5 minutes timeout for long-running requests
# Use Uvicorn workers for FastAPI (ASGI)
worker_class = "uvicorn.workers.UvicornWorker"
preload_app = True  # Preload the application to share the ML model

# NOTE: The application (`webservice:app`) should be specified when running gunicorn,
# e.g., `gunicorn -c webservice_prod.py webservice:app`
# The `if __name__ == "__main__":` block has been removed as it's not standard
# for a gunicorn config file and assumes direct execution.
