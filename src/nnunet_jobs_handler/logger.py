import logging
import os
from datetime import datetime

from config import config

# log dir
log_dir = config["log_dir"] 
print(f'log_dir={log_dir}')
if not os.path.exists(log_dir):
    print('log_dir not found. So, creating one...')
    os.makedirs(log_dir)

# Define log file name with timestamp
log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
print(f'log_file={log_file}')

# Set up logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Change to logging.INFO if you don't want debug messages

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# File Handler
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log(msg):
    logger.info(msg)
    
# Function to log an exception
def log_exception(e):
    logger.error("Exception occurred", exc_info=e)

def log_and_raise_exception(e):
    logger.error("Exception occurred", exc_info=e)
    raise e

#from fastapi import Request
#def log_request(request: Request):
#    client_ip = request.client.host if request.client else "Unknown IP"
#    user_agent = request.headers.get("User-Agent", "Unknown User-Agent")
#    logger.info(f"Received request from {client_ip}, User-Agent: {user_agent}")

# Example usage:
if __name__ == "__main__":
    logger.info("Logger initialized")
    try:
        1 / 0  # Intentional error
    except Exception as e:
        log_exception(e)
