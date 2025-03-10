#!/usr/bin/env python3
"""
Production WSGI server for the webservice using gunicorn
"""
import os
import multiprocessing

# Gunicorn config
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = min(multiprocessing.cpu_count() + 1, 4)  # Max 4 workers
timeout = 300  # 5 minutes timeout for long-running requests
worker_class = "sync"  # Use sync workers for GPU processing
preload_app = True  # Preload the application to share the ML model

# Import the Flask app
from webservice import app as application

if __name__ == "__main__":
    # This block is for running with gunicorn directly
    import sys
    from gunicorn.app.wsgiapp import WSGIApplication
    
    sys.argv = ["gunicorn", "webservice:app", 
                f"--bind={bind}", 
                f"--workers={workers}", 
                f"--timeout={timeout}", 
                f"--worker-class={worker_class}",
                "--preload"]
    
    WSGIApplication().run()
