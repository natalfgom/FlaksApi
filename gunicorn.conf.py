import os

bind = "0.0.0.0:" + os.environ.get("PORT", "8000")
timeout = 600
workers = 4
worker_class = "sync" 